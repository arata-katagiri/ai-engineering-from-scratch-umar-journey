# 1.

# from datasets import load_dataset

# ds = load_dataset("nyu-mll/glue", "mrpc", split="train")

# for i in range(5):
#     print(ds[i])

# 2.

from datasets import load_dataset
# import time

ds = load_dataset("allenai/c4", "en", split="train")

# start = time.time()
# count = 0

# for example in ds:
#     count += 1
#     if time.time() - start >= 10:
#         break

# print(f"Processed {count} examples in 10 seconds")

# 3.

# ds.to_parquet("ds_train.parquet")

# 4.

split = ds.train_test_split(test_size=0.15, seed=42)
train_val = split["train"].train_test_split(test_size=0.15, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]