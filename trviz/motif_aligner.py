import subprocess
import os
from typing import Tuple
from typing import List

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from Bio.Align.Applications import MuscleCommandline
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align.Applications import MafftCommandline
from Bio import AlignIO


class MotifAligner:

    def align(self,
              sample_ids: List[str],
              encoded_vntrs: List[str],
              vid: str = None,
              output_dir: str = None,
              tool: str = "mafft",
              ) -> Tuple[List, List]:
        """
        Align encoded VNTRs using multiple sequence alignment tools. Default tool is MAFFT.

        :param sample_ids: sample ids
        :param encoded_vntrs: encoded tandem repeats
        :param tool: the tool name for multiple sequence alignment (options: MAFFT (default))
        :param output_dir: base directory for output file
        :param vid: ID for the tandem repeat
        """
        motif_aligner = self._get_motif_aligner(tool)
        return motif_aligner(sample_ids, encoded_vntrs, vid, output_dir)

    def _get_motif_aligner(self, tool):
        if tool == 'mafft':
            return self._align_motifs_with_mafft
        elif tool == 'muscle':
            return self._align_motifs_with_muscle
        elif tool == 'clustalo':
            return self._align_motifs_with_clustalo
        else:
            ValueError(tool)

    @staticmethod
    def _align_motifs_with_muscle(sample_ids, labeled_vntrs, vid, output_dir):
        muscle_cline = MuscleCommandline('muscle', clwstrict=True)
        data = '\n'.join(['>%s\n' % str(sample_ids[i]) + labeled_vntrs[i] for i in range(len(labeled_vntrs))])
        stdout, stderr = muscle_cline(stdin=data)
        alignment = AlignIO.read(StringIO(stdout), "clustal")
        aligned_vntrs = [str(aligned.seq) for aligned in alignment]

        return sample_ids, aligned_vntrs  # TODO sample_ids are not correctly sorted

    @staticmethod
    def _align_motifs_with_clustalo(sample_ids, labeled_vntrs, vid, output_dir):
        clustalo_cline = ClustalOmegaCommandline('clustalo', infile="data.fasta", outfile="test.out",
                                                 force=True,
                                                 clusteringout="cluster.out")

        # Use dist-in (and use pre computed distance)
        # See the clusters - and plot only for those clusters.

        data = '\n'.join(['>%s\n' % str(sample_ids[i]) + labeled_vntrs[i] for i in range(len(labeled_vntrs))])
        stdout, stderr = clustalo_cline()
        alignment = AlignIO.read(StringIO(stdout), "clustal")
        aligned_vntrs = [str(aligned.seq) for aligned in alignment]

        return sample_ids, aligned_vntrs  # TODO sample_ids are not correctly sorted

    def _align_motifs_with_mafft(self, sample_ids, labeled_vntrs, vid, output_dir, preserve_order=False):
        temp_input_name = f"{output_dir}/alignment_input.fa"
        temp_output_name = f"{output_dir}/alignment_output.fa"
        if vid is not None:
            temp_input_name = f"{output_dir}/alignment_input_{vid}.fa"
            temp_output_name = f"{output_dir}/alignment_output_{vid}.fa"

        data = '\n'.join(['>%s\n' % sample_ids[i] + labeled_vntrs[i] for i in range(len(labeled_vntrs))])
        with open(temp_input_name, "w") as f:
            f.write(data)

        # TODO call mafft using pysam wrapper (
        # mafft_exe = "/usr/bin/mafft"
        # mafft_cline = MafftCommandline(mafft_exe, input=temp_input_name)

        if preserve_order:
            os.system("mafft --quiet --text --auto {} > {}".format(temp_input_name, temp_output_name))
        else:
            os.system("mafft --quiet --text --auto --reorder {} > {}".format(temp_input_name, temp_output_name))

        aligned_vntrs = []
        sample_ids = []
        tr_seq = None
        # Check if the output file exist, if not, raise error (mafft doesn't work)
        with open(temp_output_name, "r") as f:
            for line in f:
                if line.startswith(">"):
                    sample_ids.append(line.strip()[1:])
                    if tr_seq is not None:
                        aligned_vntrs.append(tr_seq)
                    tr_seq = ""
                else:
                    tr_seq += line.strip()
        if len(tr_seq) > 0:
            aligned_vntrs.append(tr_seq)

        if len(aligned_vntrs) == 0:
            raise ValueError(f"No aligned VNTRs in {temp_output_name} file")

        return sample_ids, aligned_vntrs
