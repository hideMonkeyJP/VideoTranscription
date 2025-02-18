from setuptools import setup, find_packages

setup(
    name="video_transcription",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ffmpeg-python>=0.2.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.9",
) 