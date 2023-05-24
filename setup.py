from setuptools import setup, find_packages

# setup(
#     name="demo-jformat",
#     version="0.0.1",
#     description="Reformats files to stdout",
#     install_requires = ["click", "colorama"],
#     entry_points="""
#     [console_scripts]
#     jformat=src.Automation_with_Command_Line_Tools.Creating_Single_File_Script.python_cli:main
#     """,
#     author="sepideh Hosseinian",
#     author_email="s.sepideh.hoseinian@gmail.com",
#     packages=find_packages(),
# )

with open('requirements.txt', 'r') as _f:
    requirements = [line for line in _f.read().split('\n')]

setup(
    name='summarize',
    description='demo python CLI tool to summarize text using HuggingFace',
    packages=find_packages(),
    author='Sepideh Hosseinian',
    entry_points="""
    [console_scripts]
    summarize=src.Automation_with_Command_Line_Tools.Summarization.summarize:main
    """,
    install_requires=requirements,
    version='0.0.1',
    url='https://github.com/alfredodeza/huggingface-summarization',
)