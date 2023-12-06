def label2class(label):
    if label[0] == 1:
        return "high"
    else:
        if label[2] == 1:
            return "medium"
        elif label[1] == 1:
            return "low"
        else:
            return "normal"