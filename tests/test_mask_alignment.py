import h5py
import numpy as np
import pytest
from scripts.audit_data import load_mat

def write_fixture(path, label=2, invalid=False):
    image=np.arange(20,dtype=np.int16).reshape(4,5)
    mask=np.zeros((4,5),dtype=np.uint8);mask[1,3]=1
    with h5py.File(path,'w') as f:
        g=f.create_group('cjdata');g['image']=image.T
        g['tumorMask']=mask if invalid else mask.T
        g['PID']=np.array([ord(x) for x in 'patient_A'],dtype=np.uint16).reshape(-1,1)
        g['label']=np.array([[label]])
    return image,mask

def test_matlab_coordinate_alignment_and_mapping(tmp_path):
    path=tmp_path/'sample.mat';expected,mask=write_fixture(path)
    image,actual,pid,label=load_mat(path)
    np.testing.assert_array_equal(image,expected)
    np.testing.assert_array_equal(actual,mask)
    assert image[actual.astype(bool)].item()==8
    assert pid=='patient_A' and label=='glioma'

def test_misaligned_mask_rejected(tmp_path):
    path=tmp_path/'sample.mat';write_fixture(path,invalid=True)
    with pytest.raises(ValueError):load_mat(path)

def test_unknown_class_rejected(tmp_path):
    path=tmp_path/'sample.mat';write_fixture(path,label=4)
    with pytest.raises(ValueError):load_mat(path)
