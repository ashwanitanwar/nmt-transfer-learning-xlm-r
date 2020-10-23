from conllu import parse

test_or_train = "train"
dep_par_path = "./data/raw/hi/hi_hdtb-ud-" + test_or_train + ".conllu"
dep_par_file = open(dep_par_path, "r", encoding="utf-8")

output_path_dep_par = "./data/processed/hi/" + test_or_train + ".txt"
sentences = parse(dep_par_file.read())

total_sentences=0
total_tokens=0

with open(output_path_dep_par, "w+", encoding="utf-8") as out_file_dep_par:
    for sentence in sentences:
        for token in sentence:
            form = token["form"]
            head = token["head"]
            deprel = token["deprel"]
            out_file_dep_par.write(form + " " +str(head) + "-" + deprel + "\n")
            total_tokens += 1
        out_file_dep_par.write("\n")
        total_sentences += 1

print("Total sentences processed:", total_sentences, "Total tokens processed:", total_tokens)
dep_par_file.close()
