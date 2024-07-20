from setuptools import setup

setup(
    name="localaichat",
    packages=["localaichat"],  # this must be the same as the name above
    version="0.0.1",
    description="A Python package for easily interfacing with local llms, with robust features and minimal code complexity.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="psych0v0yager",
    author_email="psych0v0yager@gmail.com",
    url="https://github.com/psych0v0yager/simpleaichat",
    keywords=["chatgpt", "openai", "text generation", "ai", "local LLMs"],
    classifiers=[],
    license="MIT",
    entry_points={
        "console_scripts": ["localaichat=localaichat.cli:interactive_chat"]
    },
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
        "fire>=0.3.0",
        "httpx>=0.24.1",
        "python-dotenv>=1.0.0",
        "orjson>=3.9.0",
        "rich>=13.4.1",
        "python-dateutil>=2.8.2",
    ],
)
