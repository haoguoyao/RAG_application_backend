

if __name__ == "__main__":
    from build_index import build_chroma_index
    from parse_document import parse_pdf

    pdf_file = "uploads/RAG_target_file.pdf"
    docs = parse_pdf(pdf_file, chunk_size=1000000, overlap=100)

    # Build the index
    index = build_chroma_index(docs)

    # Query it
    query_engine = index.as_query_engine(similarity_top_k=5,streaming=True)
    question = "What is the main contribution of this paper?"



    streaming_response = query_engine.query(question)
    # streaming_response.print_response_stream()
    # 逐步读取流式输出
    for chunk in streaming_response.response_gen:
        print(chunk, end="", flush=True)  # 实时输出