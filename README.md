💼 AI-Enhanced Resume Screening & Selection System
An HR-ready Resume Screening System that automates the process of shortlisting candidates from hundreds of resumes using modern Natural Language Processing (NLP), Regex, Fuzzy Matching, and Python automation.
Designed with a clean, intuitive Streamlit UI to support real-world recruitment workflows.

🚀 Key Features
🔍 Intelligent Resume Filtering
Match candidates against required skills using NLP & Fuzzy Matching.

📂 Bulk Resume Upload
Supports uploading .zip files of .pdf resumes for batch processing.

🧠 Regex & NLP-Powered Parsing
Automatically extracts key candidate details like Name, Email, Phone using the re module and NLP pipelines.

✅ One-Click Shortlist Download
Download all shortlisted resumes in .zip format for next-stage processing.

📈 Excel Output for Final Review
Generates an Excel sheet with structured data (Name, Email, Phone) from shortlisted resumes.

⚙️ Scalable & Modular Architecture
Built with extendability in mind — ready for integration with Firebase, databases, or AI modules.

🧬 Planned GenAI Integration
Upcoming GPT-based enhancements for:

Resume summarization (strengths + red flags)

Cover letter generation

Intelligent candidate ranking using OpenAI embeddings

💡 Real-World Impact
🎯 Inspired by actual hiring scenarios — this tool reduces hours of manual screening to a few minutes, boosting recruiter productivity and ensuring skill-based selection.

“Instead of manually reading 300 resumes, our system shortlists candidates in seconds — saving time, reducing bias, and enabling smarter hiring.”

🛠️ Tech Stack
Component	Tool/Library
UI	Streamlit
NLP & Regex	re, Word2Vec, FuzzyWuzzy
Resume Parsing	PyPDF2 / pdfplumber
Resume Matching	Word Embeddings (GloVe/Twitter100d)
Output	.zip, .xlsx files
Future AI	OpenAI API, GPT-4, LangChain

🙌 Acknowledgements
Word Embeddings: GloVe-Twitter100d

Resume parsing ideas: Inspired by real HR scenarios

GPT API (coming soon): Powered by OpenAI
