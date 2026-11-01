from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

# markdown= """
# # Project Name: Smart Student Tracker

# A simple Python-based project to manage and track student data, including their grades, age, and academic status.


# ## Features

# - Add new students with relevant info
# - View student details
# - Check if a student is passing
# - Easily extendable class-based design


# ## 🛠 Tech Stack

# - Python 3.10+
# - No external dependencies


# ## Getting Started

# 1. Clone the repo  
#    ```bash
#    git clone https://github.com/your-username/student-tracker.git

# """

code= """from langchain_text_splitters import RecursiveCharacterTextSplitter

text=Space exploration has led to incredible scientific discoveries.
From landing on the Moon to exploring Mars, humanity
continues to push the boundaries of what’s possible beyond our
planet.
These missions have not only expanded our knowledge of the
universe but have also contributed to advancements in
technology here on Earth. Satellite communications, GPS, and
even certain medical imaging techniques trace their roots back
to innovations driven by space programs

splitter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=""
)

chunk=splitter.split_text(text)

print(len(chunk[0]))
print(type(chunk))"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
    # separator=""
)

chunk=splitter.split_text(code)
print(chunk)