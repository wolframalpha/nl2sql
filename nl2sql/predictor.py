import torch
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SequencePredictor():
    def __init__(self, model, datatransformer):
        self.model = model
        self.datatransformer = datatransformer
        self.model.eval()

    @classmethod
    def fromFile(cls, model_filepath, datatransformer_filepath, *args, **kwargs):
        datatransformer = pickle.load(datatransformer_filepath)
        model = torch.load(model_filepath)
        return cls(model, datatransformer, *args, **kwargs)


    def predict(self, questions_words, columns_words):
        questions = self.datatransformer.transform_questions(questions_words)
        columns = self.datatransformer.transform_columns(columns_words)
        questions = torch.tensor(questions, dtype=torch.long, device=device)
        columns = torch.tensor(columns, dtype=torch.long, device=device)
        pred_seq = self.model(questions, columns)
        return self.datatransformer.reverse_label_sequence(pred_seq, questions=questions_words)