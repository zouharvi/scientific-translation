def text_to_vocab_simple(text):
    from nltk.tokenize import word_tokenize
    text = word_tokenize(text.lower())
    return set(text)


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


def pdf_to_vocab(pdf_url, start_keyword=None):
    # import only when necessary
    from tika import parser

    # may error out on 404
    try:
        data = parser.from_file(pdf_url)
    except:
        return set()

    data = data["content"].lower()
    if start_keyword is not None and start_keyword in data:
        index_start_keyword = data.index(start_keyword)
        # if it's too early then it's not the keyword we're looking for
        if index_start_keyword / len(data) >= 0.25:
            print(
                f"ISSUE with keyword: {index_start_keyword/len(data):.0%}, {index_start_keyword}, {len(data)}"
            )
            pass
        else:
            data = data[index_start_keyword:]
    return text_to_vocab_simple(data)


def json_dump(filename, obj, indent=2):
    import json
    with open(filename, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def json_dumpl(filename, obj):
    import json
    with open(filename, "w") as f:
        for l in obj:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def json_readl(filename):
    import json
    with open(filename, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    return data


def json_read(filename):
    import json
    with open(filename, "r") as f:
        return json.load(f)
