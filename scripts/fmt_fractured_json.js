import { promises as fs } from 'fs';
import { Formatter, FracturedJsonOptions } from 'fracturedjsonjs';

const filePaths = process.argv.slice(2);

if (filePaths.length === 0) {
  console.error('error: no file paths provided.');
  process.exit(1);
}

const formatter = new Formatter();
formatter.Options = FracturedJsonOptions.Recommended();
formatter.Options.MaxTotalLineLength = 100;
formatter.Options.MaxInlineComplexity = 2;
formatter.Options.MaxTableRowComplexity = -1;

const formatFile = async (filePath) => {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const formattedContent = formatter.Reformat(content);
    await fs.writeFile(filePath, formattedContent, 'utf-8');
    console.log(`success: formatted ${filePath}`);
  } catch (error) {
    console.error(`error: failed to format ${filePath}: ${error.message}`);
  }
};

await Promise.all(filePaths.map(formatFile));