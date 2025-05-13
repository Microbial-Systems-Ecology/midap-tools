from setuptools import setup, find_packages

def get_requirements():
    return [
        "jupyterlab",
        "numpy",
        "pandas",
        "h5py",
        "opencv-python",
        "matplotlib",
        "scipy",
        "statsmodels",
        "pytest",
        "scikit-learn",
        "plotnine"
    ]

setup(
    name="midap-tools",
    version="0.0.1",
    description="A post processing package for midap data.",
    long_description=(
        "# MIDAP-tools: A post processing package for midap data\n\n"
        "MIDAP-tools is ...\n\n"
        "Documentation at https://github.com/Microbial-Systems-Ecology/midap-tools/wiki"
    ),
    long_description_content_type="text/markdown",
    author="von Ziegler, Lukas",
    author_email="lukas.vonziegler@ethz.ch",
    python_requires=">=3.10, <4",
    url="https://github.com/Microbial-Systems-Ecology/midap-tools",
    download_url="https://github.com/Microbial-Systems-Ecology/midap-tools/archive/refs/tags/v0.0.1.tar.gz",
    keywords="Segmentation, Tracking, Biology",
    install_requires=get_requirements(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    project_urls={
        "Documentation": "https://github.com/Microbial-Systems-Ecology/midap-tools/wiki",
        "Source": "https://github.com/Microbial-Systems-Ecology/midap-tools",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)