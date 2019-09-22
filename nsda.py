import os
import os.path as op
import glob
import nibabel as nb
import numpy as np
import pandas as pd

import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import urllib.request
import zipfile
from pycocotools.coco import COCO


class NSDAccess(object):
    """
    Little class that provides easy access to the NSD data, see [http://naturalscenesdataset.org](their website)
    """

    def __init__(self, nsd_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsd_folder = nsd_folder
        self.ppdata_folder = op.join(self.nsd_folder, 'nsddata', 'ppdata')
        self.nsddata_betas_folder = op.join(
            self.nsd_folder, 'nsddata_betas', 'ppdata')

        self.behavior_file = op.join(
            self.ppdata_folder, '{subject}', 'behav', 'responses.tsv')
        self.stimuli_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
        self.stimuli_description_file = op.join(
            self.nsd_folder, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv')

        self.coco_annotation_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', '{}_{}.json')

    def download_coco_annotation_file(self, url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'):
        """download_coco_annotation_file downloads and extracts the relevant annotations files

        Parameters
        ----------
        url : str, optional
            url for zip file containing annotations, by default 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        """
        print('downloading annotations from {}'.format(url))
        filehandle, _ = urllib.request.urlretrieve(url)
        zip_file_object = zipfile.ZipFile(filehandle, 'r')
        zip_file_object.extractall(path=op.split(
            op.split(self.coco_annotation_file)[0])[0])

    def read_betas(self, subject, session_index, trial_index=[], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage'):
        """read_betas read betas from MRI files

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        session_index : int
            which session, counting from 0
        trial_index : list, optional
            which trials from this session's file to return, by default [], which returns all trials
        data_type : str, optional
            which type of beta values to return from ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR', 'restingbetas_fithrf'], by default 'betas_fithrf_GLMdenoise_RR'
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'

        Returns
        -------
        numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
            the requested per-trial beta values
        """
        data_folder = op.join(self.nsddata_betas_folder,
                              subject, data_format, data_type)
        si_str = str(session_index).zfill(2)
        if data_format == 'fsaverage':
            session_betas = []
            for hemi in ['lh', 'rh']:
                hdata = nb.load(op.join(
                    data_folder, f'{hemi}.betas_session{si_str}.mgz')).get_data()
                session_betas.append(hdata)
            out_data = np.squeeze(np.vstack(session_betas))
        else:
            out_data = nb.load(op.join(data_folder, f'betas_session{si_str}.nii.gz').get_data()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]

    def read_mapper_results(self, subject, mapper='prf', data_type='angle', data_format='fsaverage'):
        """read_mapper_results [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        mapper : str, optional
            first part of the mapper filename, by default 'prf'
        data_type : str, optional
            second part of the mapper filename, by default 'angle'
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'

        Returns
        -------
        numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
            the requested mapper values
        """
        if data_format == 'fsaverage':
            # unclear for now where the fsaverage mapper results would be
            # as they are still in fsnative format now.
            raise NotImplementedError('no mapper results in fsaverage present for now')
        else: # is 'func1pt8mm' or 'func1mm'
            ipf = op.join(self.ppdata_folder, subject, data_format, f'{mapper}_{data_type}.nii.gz')
            return nb.load(ipf).get_data()

    def read_atlas_results(self, subject, atlas='HCP_MMP1', data_format='fsaverage'):
        """read_atlas_results [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
            for surface-based data formats, subject should be the same as data_format.
            for example, for fsaverage, both subject and data_format should be 'fsaverage'
            this requires a little more typing but makes data format explicit
        atlas : str, optional
            which atlas to read,
            for volume formats, any of ['HCP_MMP1', 'Kastner2015', 'nsdgeneral', 'visualsulc'] for volume,
            for fsaverage
            can be prefixed by '.lh' or '.rh' for hemisphere-specific atlases in volume
            for surface: takes both hemispheres by default, instead when prefixed by '.rh' or '.lh'.
            By default 'HCP_MMP1'.
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm', 'MNI'], by default 'fsaverage'

        Returns
        -------
        numpy.ndarray, 1D/2D (surface) or 3D/4D (volume data formats)
            the requested atlas values
        """
        if data_format not in ('func1pt8mm', 'func1mm', 'MNI'):
            # if surface based results by exclusion
            if atlas[:3] in ('rh.', 'lh.'): # check if hemisphere-specific atlas requested
                ipf = op.join(self.ppdata_folder, 'freesurfer', subject, 'label', f'{atlas}.mgz')
                return nb.load(ipf).get_data()
            else: # more than one hemisphere requested
                session_betas = []
                for hemi in ['lh', 'rh']:
                    hdata = nb.load(op.join(
                        self.ppdata_folder, 'freesurfer', subject, 'label', f'{hemi}.{atlas}.mgz')).get_data()
                    session_betas.append(hdata)
                out_data = np.squeeze(np.vstack(session_betas))
                return out_data
        else: # is 'func1pt8mm', 'MNI', or 'func1mm'
            ipf = op.join(self.ppdata_folder, subject, data_format, 'roi', f'{atlas}.nii.gz')
            return nb.load(ipf).get_data()

    def list_atlases(self, subject, data_format='fsaverage', abs_paths=False):
        """list_atlases [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
            for surface-based data formats, subject should be the same as data_format.
            for example, for fsaverage, both subject and data_format should be 'fsaverage'
            this requires a little more typing but makes data format explicit
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm', 'MNI'], by default 'fsaverage'

        Returns
        -------
        list
            collection of absolute path names to
        """
        if data_format not in ('func1pt8mm', 'func1mm', 'MNI'):
            atlas_files = glob.glob(op.join(self.ppdata_folder, 'freesurfer', subject, 'label', '*.mgz'))
        else:
            atlas_files = op.join(self.ppdata_folder, subject, data_format, 'roi', '*.nii.gz')
        print('Atlases: found {} in {}'.format([op.split(f)[1] for f in atlas_files], op.split(atlas_files[0])[0]))
        if abs_paths:
            return [op.split(f)[1] for f in atlas_files]
        else:
            return atlas_files

    def read_behavior(self, subject, session_index, trial_index=[]):
        """read_behavior [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        session_index : int
            which session, counting from 0
        trial_index : list, optional
            which trials from this session's behavior to return, by default [], which returns all trials

        Returns
        -------
        pandas DataFrame
            DataFrame containing the behavioral information for the requested trials
        """

        behavior = pd.read_csv(self.behavior_file.format(
            subject=subject), delimiter='\t')

        # the behavior is encoded per run.
        # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
        # i.e. using indexing per session, and not using the 'run' information.
        session_behavior = behavior[behavior['SESSION'] == session_index]

        if len(trial_index) == 0:
            trial_index = slice(0, len(session_behavior))

        return session_behavior.iloc[trial_index]

    def read_images(self, image_index, show=False):
        """read_images reads a list of images, and returns their data

        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return
        show : bool, optional
            whether to also show the images, by default False

        Returns
        -------
        numpy.ndarray, 3D
            RGB image data
        """
        sf = h5py.File(self.stimuli_file, 'r')
        sdataset = sf.get('imgBrick')
        if show:
            f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
            if len(image_index) == 1:
                ss = [ss]
            for s, d in zip(ss, sdataset[image_index]):
                s.axis('off')
                s.imshow(d)
        return sdataset[image_index]

    def read_image_coco_info(self, image_index, info_type='captions', show_annot=False, show_img=False):
        """image_coco_info returns the coco annotations of a given single image

        Parameters
        ----------
        image_index : integer
            index to image in 73k format
        info_type : str, optional
            what type of annotation to return, from ['captions', 'person_keypoints', 'instances'], by default 'captions'
        show_annot : bool, optional
            whether to show the annotation, by default False
        show_img : bool, optional
            whether to show the image (from the nsd formatted data), by default False

        Returns
        -------
        coco Annotation
            coco annotation, to be used in subsequent analysis steps
        """
        if not hasattr(self, 'stim_descriptions'):
            self.stim_descriptions = pd.read_csv(
                self.stimuli_description_file, index_col=0)

        subj_info = self.stim_descriptions.iloc[image_index]

        # checking whether annotation file for this trial exists.
        # This may not be the right place to call the download, and
        # re-opening the annotations for all images separately may be slowing things down
        # however images used in the experiment seem to have come from different sets.
        annot_file = self.coco_annotation_file.format(
            info_type, subj_info['cocoSplit'])
        print('getting annotations from ' + annot_file)
        if not os.path.isfile(annot_file):
            print('annotations file not found')
            self.download_coco_annotation_file()

        coco = COCO(annot_file)
        coco_annot_IDs = coco.getAnnIds([subj_info['cocoId']])
        coco_annot = coco.loadAnns(coco_annot_IDs)

        if show_img:
            self.read_images([image_index], show=True)

        if show_annot:
            # still need to convert the annotations (especially person_keypoints and instances) to the right reference frame,
            # because the images were cropped. See image information per image to do this.
            coco.showAnns(coco_annot)

        return coco_annot
