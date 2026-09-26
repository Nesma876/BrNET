from src.evaluation.patient_metrics import metrics

def slice_metrics(y,p):
    return {'population':'slice-level performance on images from held-out patients','inference_unit':'patient; no independent slice confidence intervals',**metrics(y,p)}
