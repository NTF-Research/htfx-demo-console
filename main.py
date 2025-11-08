import os
from utils.io.EnhanceIO import get_select, get_text

from framework.HybridTaxonomyFramework import HybridTaxonomyFramework



def probability(fx : HybridTaxonomyFramework):
    while True:
        text = get_text("\nEnter text to test probability: ")
        if text in ["0","x"]:
            break

        probs = fx.proba(text)

        for label_id, prob in probs.items():
            print(f"{label_id} - {prob}")

        # proba = fx.probability([text])
        # print(proba)
        pass
    pass


def recommend(fx : HybridTaxonomyFramework):
    while True:
        text = get_text("\nEnter text to recommend: ")
        if text in ["0","x"]:
            break

        items = fx.recommend(text)
        pass
    pass

def main():
    # 
    current_dir = os.getcwd()
    workspace = f"{current_dir}/_workspaces/all-MiniLM-L6-v2_lr_faiss"
    # 
    fx = HybridTaxonomyFramework()
    fx.setup.workspace = workspace

    fx.data.setup.sqlite_attach_file = "/mnt/d/Workspaces/NTF-Solutions/HTFX/_outputs/amazon/extracts/Amazon Products.db"

    fx.data.setup.workspace = workspace
    fx.labeler.setup.workspace = workspace
    fx.embedder.setup.workspace = workspace
    fx.classifier.setup.workspace = workspace


    fx.initialize()

    options = []
    options.append("Init workspace,init")
    options.append("Labelling,labelling")
    options.append("Finetune embedder,finetune")
    options.append("Embedding,embed")
    options.append("Split data,split")
    options.append("Train,train")
    options.append("Test,test")
    options.append("Probability,probability")
    options.append("Recommend,recommend")
    while True:
        select = get_select("\nSelect option to run:", options)

        if select is None:
            continue

        if select in ["0","x"]:
            break

        if select == "init":
            fx.initialize()
            pass

        elif select == "labelling":
            fx.labelling()
            pass

        elif select == "finetune":
            fx.finetune()
            pass

        elif select == "embed":
            fx.embedding()
            pass

        elif select == "split":
            fx.split_data()
            pass

        elif select == "train":
            fx.train()
            pass

        elif select == "test":
            fx.test()
            pass

        elif select == "probability":
            probability(fx)
            pass

        elif select == "recommend":
            recommend(fx)
            pass

        else:
            print(select)


    pass

if __name__ == "__main__":
    main()