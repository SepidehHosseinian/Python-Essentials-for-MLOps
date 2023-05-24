from setuptools import setup, find_packages

setup(
    name="demo-jformat",
    version="0.0.1",
    description="Reformats files to stdout",
    install_requires = ["click", "colorama"],
    entry_points="""
    [console_scripts]
    jformat=src.Automation_with_Command_Line_Tools.Creating_Single_File_Script.python_cli:main
    """,
    author="sepideh Hosseinian",
    author_email="s.sepideh.hoseinian@gmail.com",
    packages=find_packages(),
)