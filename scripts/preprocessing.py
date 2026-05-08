import os
import re
import random
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

# Config
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Utils
def clean_caption(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_diff_word(src, trg):
    # Enforce EXACTLY one-word difference
    src_w = src.split()
    trg_w = trg.split()
    
    if len(src_w) != len(trg_w):
        return None, None

    diff_src = [w for w in src_w if w not in trg_w]
    diff_tgt = [w for w in trg_w if w not in src_w]

    if len(diff_src) != 1 or len(diff_tgt) != 1:
        return None, None

    return diff_src[0], diff_tgt[0]


# Load labels
def load_labels(labels_dir):
    records = []

    for fname in os.listdir(labels_dir): # list of all files
        if not fname.endswith("_detect_all.txt"):
            continue

        # dress_train_detect_all.txt => category_split
        category, split, *_ = fname.split("_")

        df = pd.read_csv(
            os.path.join(labels_dir, fname),
            sep="\t",
            names=["image_path", "score", "caption"],
            dtype={"image_path": str}
        )

        for _, row in df.iterrows():
            #if row["score"] < -1.0:
                #continue  
            records.append({
                "image_path": row["image_path"],
                "caption": clean_caption(row["caption"]),
                "category": category,
                "split": split,
                "score": row['score']
            })

    return pd.DataFrame(records)


# Triplet construction (Fashion200K)
def build_triplets(df, max_triplets_per_caption=6):
    
    #df MUST contain a single category only (skirt or pants or jacket or dress or top)
    
    triplets = []

    # caption -> list of image indices 
    # قائمة بأرقام الصفوف التي تحمل هذا الكابشن 
    caption2imgids = defaultdict(list)
    
    for i, row in df.iterrows():
        caption2imgids[row["caption"]].append(i) #'black dress': [0,1,12]

    # caption2imgids => ({'black dress': [0, 1, 12], 'red dress midi': [2, 15],... })
    
    # one representative image per caption
    caption2repimg = {}

    for caption, ids in caption2imgids.items():
        best_id = max(ids, key=lambda i: df.loc[i, "score"])
        caption2repimg[caption] = best_id # best image for caption {'caption': id}
        
    # caption2repimg => ({'white linen sleeveless tee': 161151, 'multicolor graphic raglan tee': 161152,... })

    # build parent captions (remove one word)
    #parent -> key  childreen-> قائمة الكابشن التي تختلف بكلمة واحدة
    parent2children = defaultdict(list)
    for caption in caption2imgids:
        words = caption.split() # "red dress midi"
        for i in range(len(words)):
            parent = " ".join(words[:i] + words[i+1:]) #" dress midi,red midi, red dress"
            parent2children[parent].append(caption)
            
    # parent2children => ({'linen sleeveless tee': ['white linen sleeveless tee', 'blue linen sleeveless tee',...],...})
           
    
    # captions that have at least one valid sibling
    modifiable = {
        c for children in parent2children.values()
        if len(children) >= 2
        for c in children
    }
    # modifiable => {'white linen sleeveless tee', 'blue linen sleeveless tee',...}
    
    #build triplets 
    for caption in modifiable: # "black dress"
        used = 0
        words = caption.split()

        #caption => black zigzag pattern trousers
        """
        parent zigzag pattern trousers
        siblings []
        parent black pattern trousers
        siblings []
        parent black zigzag trousers
        siblings ['black lurex zigzag trousers']
        src_w,tgt_w pattern lurex
        parent black zigzag pattern
        siblings []
        """

        for i in range(len(words)):
            parent = " ".join(words[:i] + words[i+1:]) # "beaded strap  gown",
            # ['blue beaded mesh  gown', 'blue beaded waist  gown', 'blue beaded blouson  gown']
            siblings = [
                c for c in parent2children[parent]
                if c != caption
            ]
            random.shuffle(siblings)
            #parent pink chiffon gown
            #siblings ['pink womens chiffon gown', 'pink beaded chiffon gown']
           
            
            for target_caption in siblings: # "red dress"
                #target_caption pink beaded chiffon gown
                src_w, tgt_w = get_diff_word(
                    caption, target_caption
                ) 
                # src_w,tgt_w => sequined beaded
                
                
                if src_w is None:
                    continue

                triplets.append({
                    "ref_path": df.loc[caption2repimg[caption], "image_path"],
                    "ref_caption": caption,
                    "mod_text": f"replace {src_w} with {tgt_w}",
                    "trg_path": df.loc[caption2repimg[target_caption], "image_path"],
                    "trg_caption": target_caption
                })

                used += 1
                if used >= max_triplets_per_caption:
                    break

            if used >= max_triplets_per_caption:
                break

    return pd.DataFrame(triplets)


# Main
if __name__ == "__main__":
    labels_dir = "data/raw/labels"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading labels...")
    df = load_labels(labels_dir)
    print("Total images:", len(df))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    print("Building train triplets...")

    all_train_triplets = []
    all_test_triplets = []

    for category in train_df["category"].unique():
        cat_df = train_df[train_df["category"] == category]
        print(f"  Category: {category} | Images: {len(cat_df)}")

        cat_triplets = build_triplets(
            cat_df,
            max_triplets_per_caption=14
        )
        all_train_triplets.append(cat_triplets)

    train_triplets = pd.concat(
        all_train_triplets, ignore_index=True
    )

    train_triplets.to_csv(
        os.path.join(out_dir, "train_triplets.csv"),
        index=False
    )
    
    print("Building test triplets...")
    
    for category in test_df["category"].unique():
        cat_df = test_df[test_df["category"] == category]
        print(f"  Category: {category} | Images: {len(cat_df)}")

        cat_triplets = build_triplets(
            cat_df,
            max_triplets_per_caption=15
        )
        all_test_triplets.append(cat_triplets)

    test_triplets = pd.concat(
        all_test_triplets, ignore_index=True
    )

    test_triplets.to_csv(
        os.path.join(out_dir, "test_triplets.csv"),
        index=False
    )
    
    val_triplets, test_triplets = train_test_split(
        test_triplets,
        test_size=0.7,
        random_state=RANDOM_SEED
    )
    
    val_triplets.to_csv(
        os.path.join(out_dir, "val_triplets.csv"),
        index=False
    )

    
    print("Done.")
    print("Train triplets:", len(train_triplets))
    print("Test triplets:", len(test_triplets))
    print("Val triplets:", len(val_triplets))