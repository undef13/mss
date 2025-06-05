"""
Pratt Parser for the Core ML Model Intermediate Language.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# tokenizer


class TokenType(Enum):
    COMMENT = r"//.*?$|/\*.*?\*/"
    WHITESPACE = r"\s+"
    ARROW = r"->"
    LBRACE = r"\{"
    RBRACE = r"\}"
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACKET = r"\["
    RBRACKET = r"\]"
    LANGLE = r"<"
    RANGLE = r">"
    COMMA = r","
    SEMICOLON = r";"
    EQUALS = r"="
    STRING = r'"(?:\\.|[^"\\])*"'
    HEX_FLOAT = r"[+-]?0x(?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)p[+-]?\d+"
    NUMBER = r"[+-]?\d+\.\d*(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+|[+-]?\d+"
    IDENTIFIER = r"[@a-zA-Z_][@a-zA-Z0-9_]*"
    MISMATCH = r"."
    EOF = auto()


@dataclass
class Token:
    kind: TokenType
    value: str
    pos: int
    line: int
    column: int


_TOKEN_REGEX = re.compile(
    "|".join(
        f"(?P<{member.name}>{member.value})" for member in TokenType if member != TokenType.EOF
    ),
    re.MULTILINE | re.DOTALL,
)


def tokenize(code: str) -> list[Token]:
    tokens: list[Token] = []
    line_num = 1
    line_start = 0
    for mo in _TOKEN_REGEX.finditer(code):
        kind_str = mo.lastgroup
        assert kind_str is not None
        value = mo.group()
        column = mo.start() - line_start
        kind = TokenType[kind_str]

        if kind not in [TokenType.COMMENT, TokenType.WHITESPACE, TokenType.MISMATCH]:
            tokens.append(
                Token(
                    kind=kind,
                    value=value,
                    pos=mo.start(),
                    line=line_num,
                    column=column,
                )
            )
        elif kind == TokenType.MISMATCH:
            error_line_start = code.rfind("\n", 0, mo.start()) + 1
            error_line_end = code.find("\n", mo.start())
            if error_line_end == -1:
                error_line_end = len(code)
            error_line = code[error_line_start:error_line_end]
            current_error_line_num = code.count("\n", 0, mo.start()) + 1
            error_message = (
                f"unexpected character: {value!r} at line {current_error_line_num}, column {column + 1}\n"
                f"  {error_line}\n"
                f"  {' ' * column}^"
            )
            raise RuntimeError(error_message)

        newlines = value.count("\n")
        if newlines > 0:
            line_num += newlines
            line_start = mo.start() + value.rfind("\n") + 1

    tokens.append(
        Token(
            kind=TokenType.EOF,
            value="EOF",
            pos=len(code),
            line=line_num,
            column=0,
        )
    )
    return tokens


# parser


class TypeInfo: ...


@dataclass
class SimpleTypeInfo(TypeInfo):
    type_name: str


@dataclass
class TensorTypeInfo(TypeInfo):
    dtype: str
    shape: list[int | float | str]
    type_name: str = "tensor"


@dataclass
class DictTypeInfo(TypeInfo):
    key_type: TypeInfo
    value_type: TypeInfo
    type_name: str = "dict"


@dataclass
class TypedName:
    name: str
    type: TypeInfo


PrimitiveValueType = int | float | str | bool


@dataclass
class BlobFile:
    path: str | None
    offset: int | None
    is_blobfile: bool = True


@dataclass
class TensorLiteral:
    dtype: str
    shape: list[int | float | str]
    value: list[PrimitiveValueType] | PrimitiveValueType | BlobFile | None
    is_tensor_literal: bool = True


@dataclass
class DictDeclLiteral:
    key_type_decl: TypeInfo
    value_type_decl: TypeInfo
    value: dict[str, str]
    is_dict_decl_literal: bool = True


_GPV_Tuple = tuple["GeneralParsedValue", ...]

GeneralParsedValue = (
    str | int | float | bool | TensorLiteral | DictDeclLiteral | BlobFile | _GPV_Tuple | None
)


@dataclass
class Statement:
    outputs: list[TypedName]
    op_type: str
    inputs: dict[str, GeneralParsedValue]
    op_name: str
    value: TensorLiteral | None  # corresponds to 'val' attribute in op_attrs

    def dbg(self, compact: bool = False) -> str:
        return _statement_dbg(self, compact=compact)


@dataclass
class Function:
    name: str
    opset_version: str | None
    inputs: list[TypedName]
    operations: list[Statement]
    returns: list[str]


@dataclass
class Program:
    version: str
    attributes: dict[str, GeneralParsedValue] = field(default_factory=dict)
    functions: list[Function] = field(default_factory=list)


# parser


class MILParser:
    def __init__(self, tokens: list[Token]):
        self.tokens: list[Token] = tokens
        self.pos: int = 0
        # initialize with a placeholder; version will be parsed.
        self.program_data: Program = Program(version="")

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _consume(
        self, expected_kind: TokenType | None = None, expected_value: str | None = None
    ) -> Token:
        token = self.tokens[self.pos]
        if expected_kind and token.kind != expected_kind:
            raise ValueError(
                f"expected token kind {expected_kind} but got {token.kind} "
                f"('{token.value}') at pos {token.pos} (line {token.line}, col {token.column})"
            )
        if expected_value and token.value != expected_value:
            raise ValueError(
                f"expected token value '{expected_value}' but got '{token.value}' "
                f"at pos {token.pos} (line {token.line}, col {token.column})"
            )
        self.pos += 1
        return token

    def _parse_type(self) -> TypeInfo:
        token = self._peek()
        if token.value == "tensor":
            self._consume(TokenType.IDENTIFIER, "tensor")
            self._consume(TokenType.LANGLE)
            dtype_token = self._consume(TokenType.IDENTIFIER)
            dtype = dtype_token.value
            self._consume(TokenType.COMMA)
            shape = self._parse_shape()
            self._consume(TokenType.RANGLE)
            return TensorTypeInfo(dtype=dtype, shape=shape)
        elif token.value == "dict":
            self._consume(TokenType.IDENTIFIER, "dict")
            self._consume(TokenType.LANGLE)
            key_type = self._parse_type()
            self._consume(TokenType.COMMA)
            value_type = self._parse_type()
            self._consume(TokenType.RANGLE)
            return DictTypeInfo(key_type=key_type, value_type=value_type)
        else:
            return SimpleTypeInfo(type_name=self._consume(TokenType.IDENTIFIER).value)

    def _parse_type_and_name(self) -> TypedName:
        type_info = self._parse_type()
        name = self._consume(TokenType.IDENTIFIER).value
        return TypedName(name=name, type=type_info)

    def _parse_typed_list(self) -> list[TypedName]:
        items: list[TypedName] = []
        while self._peek().kind != TokenType.RPAREN and self._peek().kind != TokenType.EQUALS:
            items.append(self._parse_type_and_name())
            if self._peek().value == ",":
                self._consume(TokenType.COMMA)
            else:
                break
        return items

    def _parse_shape(self) -> list[int | float | str]:
        self._consume(TokenType.LBRACKET)
        shape_dims: list[int | float | str] = []
        while self._peek().kind != TokenType.RBRACKET:
            token = self._peek()
            if token.kind == TokenType.NUMBER:
                val_str = self._consume().value
                shape_dims.append(
                    int(val_str)
                    if "." not in val_str and "e" not in val_str.lower()
                    else float(val_str)
                )
            elif token.kind == TokenType.HEX_FLOAT:
                val_str = self._consume().value
                shape_dims.append(float.fromhex(val_str))
            elif token.kind == TokenType.IDENTIFIER:
                shape_dims.append(self._consume(TokenType.IDENTIFIER).value)
            else:
                raise ValueError(f"Unexpected token in shape: {self._peek()}")

            if self._peek().value == ",":
                self._consume(TokenType.COMMA)
        self._consume(TokenType.RBRACKET)
        return shape_dims

    def _parse_value(self) -> GeneralParsedValue:
        token = self._peek()
        if token.kind == TokenType.IDENTIFIER:
            if token.value == "tensor":
                return self._parse_tensor_literal()
            elif token.value == "dict":
                return self._parse_dict_declaration_and_literal()
            elif token.value in ["true", "false"]:
                self._consume(TokenType.IDENTIFIER)
                return token.value == "true"
            else:
                return self._consume(TokenType.IDENTIFIER).value
        elif token.kind == TokenType.STRING:
            return json.loads(self._consume(TokenType.STRING).value)
        elif token.kind == TokenType.NUMBER:
            val_str = self._consume().value
            return (
                int(val_str)
                if "." not in val_str and "e" not in val_str.lower()
                else float(val_str)
            )
        elif token.kind == TokenType.HEX_FLOAT:
            val_str = self._consume().value
            return float.fromhex(val_str)
        elif token.kind == TokenType.LPAREN:
            self._consume(TokenType.LPAREN)
            values: list[GeneralParsedValue] = []
            while self._peek().kind != TokenType.RPAREN:
                values.append(self._parse_value())
                if self._peek().value == ",":
                    self._consume(TokenType.COMMA)
                else:
                    break
            self._consume(TokenType.RPAREN)
            return tuple(values)
        else:
            raise ValueError(f"Unexpected value token: {token}")

    def _parse_tensor_literal(self) -> TensorLiteral:
        self._consume(TokenType.IDENTIFIER, "tensor")
        self._consume(TokenType.LANGLE)
        dtype = self._consume(TokenType.IDENTIFIER).value
        self._consume(TokenType.COMMA)
        shape = self._parse_shape()
        self._consume(TokenType.RANGLE)
        self._consume(TokenType.LPAREN)

        val: list[PrimitiveValueType] | PrimitiveValueType | BlobFile | None = None
        if self._peek().value == "BLOBFILE":
            self._consume(TokenType.IDENTIFIER, "BLOBFILE")
            self._consume(TokenType.LPAREN)
            blob_args = self._parse_named_arguments()
            self._consume(TokenType.RPAREN)

            path_val = blob_args.get("path")
            offset_val = blob_args.get("offset")

            extracted_path: str | None = None
            if isinstance(path_val, TensorLiteral) and isinstance(path_val.value, str):
                extracted_path = path_val.value
            elif path_val is not None:
                raise ValueError(
                    f"BLOBFILE path expected string tensor literal, got {type(path_val)}"
                )

            extracted_offset: int | None = None
            if isinstance(offset_val, TensorLiteral) and isinstance(offset_val.value, int):
                extracted_offset = offset_val.value
            elif offset_val is not None:
                raise ValueError(
                    f"BLOBFILE offset expected int tensor literal, got {type(offset_val)}"
                )
            val = BlobFile(path=extracted_path, offset=extracted_offset)

        elif self._peek().kind == TokenType.LBRACKET:
            self._consume(TokenType.LBRACKET)
            val_list: list[PrimitiveValueType] = []
            while self._peek().kind != TokenType.RBRACKET:
                parsed_simple_val = self._parse_value()
                if not isinstance(parsed_simple_val, (int, float, str, bool)):
                    raise ValueError(
                        f"unexpected complex value {parsed_simple_val} in tensor literal list. expected primitive."
                    )
                val_list.append(parsed_simple_val)
                if self._peek().value == ",":
                    self._consume(TokenType.COMMA)
                else:
                    break
            self._consume(TokenType.RBRACKET)
            val = val_list
        elif self._peek().kind != TokenType.RPAREN:
            parsed_single_val = self._parse_value()
            if not isinstance(parsed_single_val, (int, float, str, bool)):
                raise ValueError(
                    f"Unexpected complex value {parsed_single_val} as single tensor literal value. Expected primitive."
                )
            val = parsed_single_val

        self._consume(TokenType.RPAREN)
        return TensorLiteral(dtype=dtype, shape=shape, value=val)

    def _parse_dict_declaration_and_literal(self) -> DictDeclLiteral:
        self._consume(TokenType.IDENTIFIER, "dict")
        self._consume(TokenType.LANGLE)
        key_type = self._parse_type()
        self._consume(TokenType.COMMA)
        value_type = self._parse_type()
        self._consume(TokenType.RANGLE)
        self._consume(TokenType.LPAREN)
        dict_literal_value = self._parse_buildinfo_dict_content()
        self._consume(TokenType.RPAREN)
        return DictDeclLiteral(
            key_type_decl=key_type,
            value_type_decl=value_type,
            value=dict_literal_value,
        )

    def _parse_buildinfo_dict_content(self) -> dict[str, str]:
        self._consume(TokenType.LBRACE)
        items: dict[str, str] = {}
        if self._peek().value == "}":
            self._consume(TokenType.RBRACE)
            return items

        while True:
            self._consume(TokenType.LBRACE)
            key_str_token = self._consume(TokenType.STRING)
            key = json.loads(key_str_token.value)
            self._consume(TokenType.COMMA)
            val_str_token = self._consume(TokenType.STRING)
            val = json.loads(val_str_token.value)
            items[key] = val
            self._consume(TokenType.RBRACE)
            if self._peek().value == ",":
                self._consume(TokenType.COMMA)
            elif self._peek().value == "}":
                break
            else:
                raise ValueError(f"Expected COMMA or RBRACE in dict content, got {self._peek()}")
        self._consume(TokenType.RBRACE)
        return items

    def _parse_named_arguments(self) -> dict[str, GeneralParsedValue]:
        args: dict[str, GeneralParsedValue] = {}
        while self._peek().kind != TokenType.RPAREN and self._peek().kind != TokenType.RBRACKET:
            name = self._consume(TokenType.IDENTIFIER).value
            self._consume(TokenType.EQUALS)
            args[name] = self._parse_value()

            if self._peek().value == ",":
                self._consume(TokenType.COMMA)
            else:
                break
        return args

    def _parse_op_attributes(self) -> dict[str, GeneralParsedValue]:
        attrs: dict[str, GeneralParsedValue] = {}
        if self._peek().kind == TokenType.LBRACKET:
            self._consume(TokenType.LBRACKET)
            attrs = self._parse_named_arguments()
            self._consume(TokenType.RBRACKET)
        return attrs

    def _parse_statement(self) -> Statement:
        outputs = self._parse_typed_list()
        self._consume(TokenType.EQUALS)

        op_type = self._consume(TokenType.IDENTIFIER).value
        self._consume(TokenType.LPAREN)
        op_inputs = self._parse_named_arguments()
        self._consume(TokenType.RPAREN)

        op_attrs = self._parse_op_attributes()
        self._consume(TokenType.SEMICOLON)

        op_name_val = op_attrs.get("name")
        parsed_op_name: str | None = None
        if isinstance(op_name_val, TensorLiteral):
            if isinstance(op_name_val.value, str):
                parsed_op_name = op_name_val.value
            elif op_name_val.value is not None:
                parsed_op_name = str(op_name_val.value)
        elif isinstance(op_name_val, str):
            parsed_op_name = op_name_val
        assert parsed_op_name is not None

        parsed_value_attr = op_attrs.get("val")
        stmt_value: TensorLiteral | None = None
        if isinstance(parsed_value_attr, TensorLiteral):
            stmt_value = parsed_value_attr
        elif parsed_value_attr is not None:
            raise ValueError(
                f"op attribute 'val' expected TensorLiteral, got {type(parsed_value_attr)}"
            )

        return Statement(
            outputs=outputs,
            op_type=op_type,
            inputs=op_inputs,
            op_name=parsed_op_name,
            value=stmt_value,
        )

    def _parse_function(self) -> Function:
        self._consume(TokenType.IDENTIFIER, "func")
        func_name = self._consume(TokenType.IDENTIFIER).value
        opset_version: str | None = None
        if self._peek().kind == TokenType.LANGLE:
            self._consume(TokenType.LANGLE)
            opset_version = self._consume(TokenType.IDENTIFIER).value
            self._consume(TokenType.RANGLE)

        self._consume(TokenType.LPAREN)
        params = self._parse_typed_list()
        self._consume(TokenType.RPAREN)

        self._consume(TokenType.LBRACE)
        statements: list[Statement] = []
        while self._peek().value != "}":
            statements.append(self._parse_statement())
        self._consume(TokenType.RBRACE)

        self._consume(TokenType.ARROW)
        self._consume(TokenType.LPAREN)
        return_vars: list[str] = []
        while self._peek().kind != TokenType.RPAREN:
            return_vars.append(self._consume(TokenType.IDENTIFIER).value)
            if self._peek().value == ",":
                self._consume(TokenType.COMMA)
            else:
                break
        self._consume(TokenType.RPAREN)
        self._consume(TokenType.SEMICOLON)

        return Function(
            name=func_name,
            opset_version=opset_version,
            inputs=params,
            operations=statements,
            returns=return_vars,
        )

    def _parse_program_header_and_attributes(self) -> None:
        self._consume(TokenType.IDENTIFIER, "program")
        self._consume(TokenType.LPAREN)
        version_token = self._consume(TokenType.NUMBER)
        self.program_data.version = version_token.value
        self._consume(TokenType.RPAREN)

        if self._peek().kind == TokenType.LBRACKET:
            self._consume(TokenType.LBRACKET)
            attr_name = self._consume(TokenType.IDENTIFIER).value
            self._consume(TokenType.EQUALS)
            attr_value = self._parse_value()
            self.program_data.attributes[attr_name] = attr_value
            self._consume(TokenType.RBRACKET)

    def parse(self) -> Program:
        self._parse_program_header_and_attributes()
        self._consume(TokenType.LBRACE)
        while self._peek().value == "func":
            self.program_data.functions.append(self._parse_function())
        self._consume(TokenType.RBRACE)
        self._consume(TokenType.EOF)
        return self.program_data


def parse_mil_program(mil_code_str: str) -> Program:
    tokens = tokenize(mil_code_str)
    parser = MILParser(tokens)
    return parser.parse()


#
# impl Debug
#


def _statement_dbg(st: Statement, *, compact: bool = False) -> str:
    inputs = (
        (
            "["
            + (
                ",".join(
                    f"(shape={getattr(inp, 'shape', None) or ''},val={getattr(inp, 'value', None) or ''})"
                    for _, inp in st.inputs.items()
                    if hasattr(inp, "shape") or hasattr(inp, "value")
                )
            )
            + "]"
        )
        if st.inputs
        else ""
    )
    this = f"{st.value.dtype}{st.value.shape},val={st.value.value}" if st.value else ""
    outputs = (
        (
            "["
            + (
                ",".join(
                    f"{out.name}:{out.type.dtype}{out.type.shape}"  # type: ignore
                    for out in st.outputs
                )
            )
            + "]"
        )
        if st.outputs
        else ""
    )
    if compact:
        return f"{st.op_type} | {st.op_name} | in={inputs},this={this},out={outputs}"
    return f"{st.op_type:15} | {st.op_name:50} | in={inputs:5}, this={this:20}, out={outputs:10}"


def resolve_path(raw_path_str: str, model_path: Path) -> Path:
    assert raw_path_str.startswith("@model_path")
    rest = raw_path_str.removeprefix("@model_path/")
    relative = Path(rest)
    return model_path / relative
