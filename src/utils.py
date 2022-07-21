
def text_to_vocab(text):
    text = text.lower().split()
    text_new = []
    split_word = None
    for w in text:
        if split_word is not None:
            if w.endswith("-"):
                # multiple consecutive splits
                split_word += w[:-1]
            else:
                text_new.append(split_word + w)
                split_word = None

        elif w.endswith("-"):
            split_word = w[:-1]
        elif w.replace("-", "").isalpha():
            # allow any words that are composed of letters or contain dashes in between
            text_new.append(w)

    return set(text_new)


