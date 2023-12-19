def get_semantic_context(model):
    while True:
        try:
            topic = input("Get similar words in the semantic context for the given topic or enter 'q' to leave: ")
            if topic == "q":
                break
            print([tupel[0] for tupel in model.wv.most_similar(topic)])
        except:
            print("Please try again or enter 'q' to leave the program.")
