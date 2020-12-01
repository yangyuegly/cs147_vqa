import preprocess

class VQA:
    def __init__(self, annotations, questions):
        self.annotations = annotations
        self.questions = questions
        self.answers = None

def main():
    fpath_anno = "./data/mscoco_train2014_annotations.json"
    fpath_q_mc = "./data/MultipleChoice_mscoco_train2014_questions.json"
    annotations, questions = preprocess.load_text(fpath_anno, fpath_q_mc)
    vqa = VQA(annotations, questions)