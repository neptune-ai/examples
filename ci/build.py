import re

from glob import glob
from pathlib import Path
from subprocess import call


def clean_py_script(filename):
    EXCLUDED_STR = ['# In[', '#!/usr', '# coding:']
    code_text = filename.read_text().split('\n')
    lines = [line for line in code_text if all(l not in line for l in EXCLUDED_STR)]
    lines = [line.replace('# ##', '#') for line in lines]
    lines = [line.replace('# #', '#') for line in lines]
    clean_code = '\n'.join(lines)
    clean_code = re.sub(r'\n{2,}', '\n\n', clean_code)
    filename.write_text(clean_code.strip())


def nbconvert(**kwargs):
    command = """jupyter nbconvert \
        --TagRemovePreprocessor.enabled=True \
        --TagRemovePreprocessor.remove_cell_tags="{tags}" \
        --output-dir {output_dir} \
        --to {format} {notebook_filename}
    """.format(**kwargs)
    retcode = call(command, shell=True)
    if retcode:
        raise Exception('Converting {notebook_filename} to format {format} outputting into directory {output_dir} failed'.format(**kwargs))


def nbconvert_with_renaming(**kwargs):
    command = """jupyter nbconvert \
        --TagRemovePreprocessor.enabled=True \
        --TagRemovePreprocessor.remove_cell_tags="{tags}" \
        --output {output} \
        --output-dir {output_dir} \
        --to {format} {notebook_filename}
    """.format(**kwargs)
    retcode = call(command, shell=True)
    if retcode:
        raise Exception('Converting {notebook_filename} to format {format} outputting into directory {output_dir} failed'.format(**kwargs))


def build_tests(path):
    tests_dir = path.parent / 'tests'

    # create .py ipython script with fixed libraries -> you need to run it with ipython my_file.py
    nbconvert_with_renaming(
        tags=repr(['comment', 'neptune_stop', 'library_updates', 'bash_code', 'exclude']),
        output=path.stem + '_fixed_libs',
        output_dir=tests_dir,
        format="python",
        notebook_filename=path
    )
    clean_py_script(tests_dir / (path.stem + '_fixed_libs.py'))

    # create .py ipython script with upgraded libraries -> you need to run it with ipython my_file.py
    nbconvert_with_renaming(
        tags=repr(['comment', 'neptune_stop', 'bash_code', 'exclude']),
        output=path.stem + '_upgraded_libs',
        output_dir=tests_dir,
        format="python",
        notebook_filename=path
    )
    clean_py_script(tests_dir / (path.stem + '_upgraded_libs.py'))


def build_docs(path):
    docs_dir = path.parent / 'docs'

    # create notebook without tests and comments
    nbconvert(
        tags=repr(['comment','tests','library_updates','exclude']),
        output_dir=docs_dir,
        format="notebook",
        notebook_filename=path
    )

    # create .py script
    nbconvert(
        tags=repr(['comment','installation', 'neptune_stop','tests','library_updates','bash_code','exclude']),
        output_dir=docs_dir,
        format="python",
        notebook_filename=path
    )
    clean_py_script(docs_dir / (path.stem + '.py'))


def build_showcase(path):
    showcase_dir = path.parent / 'showcase'
        # create notebook without tests
    nbconvert(
        tags=repr(['tests', 'library_updates']),
        output_dir=showcase_dir,
        format="notebook",
        notebook_filename=path
    )


def build(path):
    path = Path(path)
    build_tests(path)
    build_docs(path)
    build_showcase(path)


source_files = []
source_files.extend(glob('integrations/*/*.ipynb', recursive=True))
source_files.extend(glob('product-tours/*/*.ipynb', recursive=True))
source_files.extend(glob('quick-starts/*/*.ipynb', recursive=True))

excluded_files = []
excluded_files.extend(glob('integrations/r/*.ipynb', recursive=True))
excluded_files.extend(glob('integrations\\r\\*.ipynb', recursive=True))

if __name__ == "__main__":
    for path in source_files:
        if path not in excluded_files:
            build(path)
