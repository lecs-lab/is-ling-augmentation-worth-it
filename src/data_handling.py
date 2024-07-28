"""Defines models and functions for loading, manipulating, and writing task data"""
from functools import reduce
from typing import Optional, List, Union






# def write_predictions(data: List[IGT], tokenizer, trainer: Trainer, labels, out_path):
#     """Runs predictions for a dataset and writes the output IGT"""
#     dataset = prepare_dataset(data=data, tokenizer=tokenizer, labels=labels, device='cpu')
#     preds = trainer.predict(dataset).predictions
#     decoded_preds = [[labels[index] for index in pred_seq if len(labels) > index >= 0] for pred_seq in preds]

#     with open(out_path, 'w') as outfile:
#         for line, line_preds in zip(data, decoded_preds):
#             # Write the data in the appropriate format
#             outfile.write("\\t " + line.transcription)
#             outfile.write("\n\\m " + line.segmentation)

#             # Trim preds to the number of morphemes
#             line_preds = line_preds[:len(line.morphemes())]
#             # Combine predictions into a string
#             line_pred_string = "\n\\p "
#             for pred_gloss in line_preds:
#                 if pred_gloss == "[SEP]":
#                     line_pred_string += " "
#                 elif line_pred_string[-1] == " ":
#                     line_pred_string += pred_gloss
#                 else:
#                     line_pred_string += "-" + pred_gloss

#             outfile.write(line_pred_string)
#             outfile.write("\n\\l " + line.translation + "\n\n")


# def write_igt(data: List[IGTLine], out_path):
#     with open(out_path, 'w') as outfile:
#         for line in data:
#             # Write the data in the appropriate format
#             outfile.write("\\t " + line.transcription)
#             outfile.write("\n\\m " + line.segmentation)
#             outfile.write("\n\\p " + line.glosses)
#             outfile.write("\n\\l " + line.translation + "\n\n")
