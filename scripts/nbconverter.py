import nbformat
from nbformat import read
import fire


def convert(infile: str, outfile: str = None):
    # Read the notebook
    with open(f'{infile}.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, nbformat.current_nbformat)

    # Write code cells to a Python script
    outfile = infile.split('./')[1].split('/')[1]

    with open(f'./scripts/{outfile}.py', 'w') as f:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                f.write(cell.source + '\n\n')
        print('file converted and saved in the scripts directory')


if __name__ == '__main__':
    fire.Fire(convert)

