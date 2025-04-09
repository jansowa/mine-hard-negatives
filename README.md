# Project to mine hard negatives and set scores for knowledge distillation

### How to run?
1. Create environment (TODO: add info)
2. Add indexes for each file with:
```commandline
python add_indexes.py --input_file {file_path} --output_file {output_path}
```
3. Add all documents to database:
```commandline
python add_documents_to_db.py --dataset_path {file_path_with_indexes}
```
4. Create file with hard negatives and scores:
```commandline
python find_negatives.py --dataset_path {input_file_path} --output_path {output_file_path}
```