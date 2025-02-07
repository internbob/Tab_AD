from __future__ import annotations

from typing import Optional

from heudiconv.utils import SeqInfo


def create_key(
        template: Optional[str],
        outtype: tuple[str, ...] = ("nii.gz",),
        annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(
        seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    data = create_key("run{item:03d}")
    t1w = create_key('sub-{subject}/ses-M000/anat/sub-{subject}_ses-M000_seq-{seqitem}_T1w')

    info = {t1w: []}

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        if ('T1' in s.series_description or 'MPRAGE' in s.series_description or 't1' in s.series_description or
            'mprage' in s.series_description or 'MPrage' in s.series_description or
            'MP-RAGE' in s.series_description or 'MPRage' in s.series_description) and s.dim3 > 60:
            info[t1w].append(s.series_id)
    return info
