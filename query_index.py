

if __name__ == "__main__":
    from build_index import build_chroma_index
    from parse_document import parse_pdf

    pdf_file = "uploads/RAG_target_file.pdf"
    docs = parse_pdf(pdf_file, chunk_size=1000000, overlap=100)

    # Build the index
    index = build_chroma_index(docs)

    # Query it
    query_engine = index.as_query_engine(similarity_top_k=5,streaming=True)
    questions = [
    # General Model Questions
    "What is the total number of parameters in DeepSeek-V3?",
    "How many parameters are activated per token in DeepSeek-V3?",
    "What are the primary architectures used in DeepSeek-V3?",

    # Architecture and Training
    "What is Multi-Head Latent Attention (MLA), and why is it used in DeepSeek-V3?",
    "How does DeepSeekMoE ensure load balancing without auxiliary loss?",
    "What is the purpose of Multi-Token Prediction (MTP) in DeepSeek-V3?",
    "How does DeepSeek-V3 optimize memory consumption during training?",
    "What is DualPipe, and how does it improve pipeline parallelism?",
    "What techniques are used in DeepSeek-V3 to reduce communication overhead in cross-node training?",

    # Pre-Training and Post-Training
    "What dataset size (in tokens) was used for DeepSeek-V3 pre-training?",
    "What reinforcement learning techniques are used in DeepSeek-V3 post-training?",
    "How does DeepSeek-V3 employ self-rewarding strategies?",

    # Performance and Evaluation
    "How does DeepSeek-V3 compare to GPT-4o and Claude-3.5-Sonnet in performance?",
    "On which benchmarks does DeepSeek-V3 achieve state-of-the-art results?",
    "What are the advantages of DeepSeek-V3 in mathematical reasoning tasks?",
    "How does DeepSeek-V3 perform on long-context evaluations like Needle In A Haystack?",

    # Inference and Deployment
    "What is the recommended deployment unit for DeepSeek-V3?",
    "How does DeepSeek-V3 manage load balancing during inference?",
    "What optimizations does DeepSeek-V3 use for efficient inference on large-scale clusters?",

    # Technical and Hardware Considerations
    "What are the key hardware requirements for training DeepSeek-V3?",
    "How does DeepSeek-V3 utilize FP8 training for efficiency?",
    "What are the main suggestions for future AI hardware based on DeepSeek-V3’s training needs?"
    ]



    streaming_response = query_engine.query(questions[0])
    # streaming_response.print_response_stream()
    # 逐步读取流式输出
    for chunk in streaming_response.response_gen:
        print(chunk, end="", flush=True)  # 实时输出