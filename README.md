[![Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/konard/universal-vector-search)

# Universal Sentence Encoder Multilingual Vector Search Example
This repository contains example code to demonstrate the use of the Universal Sentence Encoder Multilingual model for semantic search.

## Prerequisites
Ensure that you have the following installed:
- Python 3: If you don't have Python 3 installed, please visit the [official website](https://www.python.org/downloads/) to install it.
- pip: pip is a package manager for Python and is included by default with Python 3. If pip is not installed, it can be installed following the instructions on the [official pip installation guide](https://pip.pypa.io/en/stable/installation/).

## Installation
Follow these steps to setup your environment:

1. Install the necessary Python libraries:
    Open a terminal and run the commands:

    ```bash
    pip3 install tensorflow
    pip3 install tensorflow-hub
    pip3 install tensorflow-text
    pip3 install -U scikit-learn
    pip3 install scipy
    ```

2. Clone the repository:
    Once you have Python and pip setup, you can clone this repository to your local machine by running:

    ```bash
    git clone https://github.com/Konard/universal-vector-search
    ```

## Usage
Open a terminal, navigate to the directory where you've cloned this repository and type the following command to run the script:

```bash
python3 example.py
```

The script will then:

1. Load the Universal Sentence Encoder Multilingual model
2. Start taking user input for queries to search the predefined set of messages. The messages are ranked by their relevance to the input query.
3. To stop the script, simply press enter without typing anything when asked for a query. 

It also measures and prints the time taken to both start up (load the model and embedding) and execute each input query.

Output to console may look like this:
![Screenshot_20231214_105939](https://github.com/Konard/universal-vector-search/assets/1431904/a512c7ef-7ced-4bc8-b91a-791673b60fdd)

