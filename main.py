import os
import tkinter as tk
from tkinter import ttk
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment
APIKEY = os.getenv("OPENAI_API_KEY")

class AppleStyleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SearchDoc")
        self.root.geometry("400x300")

        title_label = ttk.Label(self.root, text="Welcome to SearchDoc!", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=20)

        self.search_entry = ttk.Entry(self.root, font=("Helvetica", 14))
        self.search_entry.pack(padx=20, fill=tk.X)

        search_button = ttk.Button(self.root, text="Search Some Docs", command=self.perform_search)
        search_button.pack(pady=10)

        self.search_results = tk.Text(self.root, font=("Helvetica", 14), bg="black", fg="white", padx=10, pady=10)
        self.search_results.pack(padx=20, fill=tk.BOTH, expand=True)

    def perform_search(self):
        try:
            query = self.search_entry.get()
            loader = TextLoader(os.path.join(os.getcwd(), "data", "data.txt"))
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(openai_api_key=APIKEY)
            docsearch = Chroma.from_documents(texts, embeddings)

            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=APIKEY), 
                chain_type="stuff", 
                retriever=docsearch.as_retriever()
            )

            qa_results = qa.run(query=query, n_results=2)
            self.search_results.delete(1.0, tk.END)

            if isinstance(qa_results, list):
                formatted_results = "\n\n".join(result.strip() for result in qa_results)
            else:
                formatted_results = str(qa_results).strip()

            self.search_results.insert(tk.END, formatted_results + "\n")
        except Exception as e:
            self.search_results.delete(1.0, tk.END)
            self.search_results.insert(tk.END, f"Error: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppleStyleApp(root)
    root.mainloop()
