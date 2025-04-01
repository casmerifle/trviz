from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import IndexLocator
import matplotlib as mpl
import numpy as np
import distinctipy
import pandas as pd

from trviz.utils import PRIVATE_MOTIF_LABEL


class TandemRepeatVisualizer:

    def __init__(self):
        self.symbol_to_color = None

    @staticmethod
    def encode_tr_sequence(labeled_motifs):
        decomposed_motifs = []
        for motif in labeled_motifs:
            encoded = []
            for m in motif:
                if m == '-':
                    encoded.append(0)
                else:
                    encoded.append(ord(m) - 64)  # 1-base start
            decomposed_motifs.append(encoded)

        return decomposed_motifs

    @staticmethod
    def _get_unique_labels(aligned_repeats):
        unique_repeats = Counter()
        for rs in aligned_repeats:
            for r in rs:
                if r != '-':
                    unique_repeats[r] += 1

        # sort by the frequency
        return list(sorted(unique_repeats, key=unique_repeats.get, reverse=True))

    @staticmethod
    def plot_motif_color_map(symbol_to_motif, motif_counter, symbol_to_color, file_name,
                             figure_size=None,
                             box_size=1,
                             box_margin=0.1,
                             label_size=None,
                             show_figure=False,
                             dpi=300):
        """ Generate a plot for motif color map """

        if box_margin < 0 or box_margin > box_size:
            raise ValueError(f"Box margin should be between 0 and {box_size}.")

        if symbol_to_color is None:
            raise ValueError("Symbol to color is not set.")

        if figure_size is not None:
            fig, ax = plt.subplots(figsize=figure_size)
        else:
            max_motif_length = len(max(symbol_to_motif.values(), key=len))
            w = len(symbol_to_motif) // 10 + 5 if len(symbol_to_motif) > 50 else len(symbol_to_motif) // 5 + 1
            h = max_motif_length // 4 + 1 if max_motif_length > 50 else max_motif_length // 3 + 1
            if h * dpi > 2 ** 16:
                h = 2 ** 15 // dpi * 0.75  # "Weight and Height must be less than 2^16"
            fig, ax = plt.subplots(figsize=(w, h))
        ax.set_aspect('equal')

        max_motif_length = len(max(symbol_to_motif.values(), key=len))
        # TODO: figure and font size
        yticklabels = []
        y_base_position = len(symbol_to_color) - 1
        has_gap_in_symbol_to_color = False
        for (symbol, color) in symbol_to_color.items():
            if symbol == '-':
                has_gap_in_symbol_to_color = True
                continue
            box_position = [box_margin, y_base_position + box_margin]
            box_width = box_size - 2 * box_margin
            box_height = box_size - 2 * box_margin
            ax.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                       linewidth=0,
                                       facecolor=color,
                                       edgecolor="white"))
            y_base_position -= 1
            motif = symbol_to_motif[symbol]
            text = f"{motif:<{max_motif_length + 2}}" + f"{motif_counter[motif]:>5}"
            yticklabels.append(text)

        ax.yaxis.tick_right()
        ax.set_ylim(top=len(symbol_to_color))
        yticks_count = len(symbol_to_color)
        if has_gap_in_symbol_to_color:
            yticks_count = len(symbol_to_color) - 1
        ax.set_yticks([y * box_size + box_size*0.5 for y in range(yticks_count)])  # minus 1 for gap, '-'
        ax.set_yticklabels(yticklabels[::-1])

        # font size
        if label_size is not None:
            ax.tick_params(axis='both', which='major', labelsize=label_size)
            ax.tick_params(axis='both', which='minor', labelsize=label_size)
        # font
        for tick in ax.get_yticklabels():
            tick.set_fontproperties("monospace")

        # Remove yticks
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            length=0)  # labels along the bottom edge are on

        # Hide xticks
        ax.set_xticks([])

        # No frame
        ax.set_frame_on(False)
        fig.savefig(file_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
        if show_figure:
            plt.show()
        plt.close(fig)

    def trplot(self,
               adjacent_methylation_flag,
               aligned_labeled_repeats: List[str],
               sample_ids: List[str],
               figure_size: Tuple[int, int] = None,
               output_name: str = None,
               dpi: int = 300,
               alpha: float = 0.6,
               box_line_width: float = 0,
               hide_xticks: bool = False,
               hide_yticks: bool = False,
               symbol_to_motif: Dict[str, str] = None,
               sort_by_clustering: bool = True,
               hide_dendrogram: bool = True,
               compression_flag = True,
               repeat_coord_start = 0,
               repeat_coord_end = 0,
               sample_to_label: Dict[str, str] = None,
               motif_marks: Dict[str, str] = None,
               allele_as_row: bool = True,
               xlabel_size: int = 9,
               ylabel_size: int = 9,
               xlabel_rotation: int = 0,
               ylabel_rotation: int = 0,
               private_motif_color: str = 'black',
               frame_on: Dict[str, bool] = None,
               show_figure: bool = False,
               no_edge: bool = False,
               color_palette: str = None,
               colormap: ListedColormap = None,
               motif_style: str = "box",
               xtick_step: int = 1,
               ytick_step: int = 1,
               xtick_offset: int = 0,
               ytick_offset: int = 0,
               title: str = None,
               xlabel: str = None,
               ylabel: str = None,
               colored_motifs: List[str] = None,
               debug: bool = False
               ):
        """
        Generate a plot showing the variations in tandem repeat sequences.
        A distinct color is assigned to each motif.
        For private motifs (with low frequency), the same color (in black by default) may be assigned.

        :param adjacent_methylation_flag: MANDATORY ARGUMENT (specify as the bool False if unsure); if demonstration of adjacent regions and their methylation is desired,
                specify the filepath of a metadata .tsv file with the following three columns:
                        (1) sample_id    (2) filepath_of_bedMethyl_file (output from ModKit2 / wf-human-variation pipeline) with only the relevant methylation regions    (3) chromosome of ROI in the format "chr1"
                a red-blue heatmap of methylation will be generated in the final trplot()
        :param aligned_labeled_repeats: aligned and encoded tandem repeat sequences.
        :param sample_ids: sample IDs
        :param figure_size: figure size
        :param output_name: output file name
        :param dpi: DPI for the plot
        :param alpha: alpha value for the plot
        :param box_line_width: line width for box edges
        :param hide_xticks: if true, hide xticks
        :param hide_yticks: if true, hide yticks
        :param symbol_to_motif: a dictionary mapping symbols to motifs
        :param sort_by_clustering: if true, sort the samples by clustering
        :param motif_marks: a dictionary mapping sample to motif marks
        :param hide_dendrogram: if true, hide the dendrogram
        :param compression_flag: if true, compresses stretches of identical STRs in the sorted repeats
        :param sample_to_label: sample to label dictionary (default is None)
        :param motif_marks: a dictionary mapping sample to motif marks
        :param allele_as_row: if true, plot allele as row (default is true)
        :param xlabel_size: x label size (default is 8)
        :param ylabel_size: y label size (default is 8)
        :param xlabel_rotation: x label rotation (default is 0)
        :param ylabel_rotation: y label rotation (default is 0)
        :param private_motif_color: the color for private motifs. Default is black
        :param frame_on: a dictionary mapping sample to frame on.
                         Default is {'top': False, 'bottom': True, 'right': False, 'left': True}
        :param show_figure: if true, show the figure
        :param no_edge: if true, hide the edge of the boxes
        :param color_palette: color palette name (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        :param colormap: Matplotlib ListedColormap object
        :param motif_style: motif style. One of "box", "arrow", "triangle"
        :param xtick_step: x tick step (default is 1)
        :param ytick_step: y tick step (default is 1)
        :param xtick_offset: x tick offset (default is 0)
        :param ytick_offset: y tick offset (default is 0)
        :param title: title for the plot
        :param xlabel: x label
        :param ylabel: y label
        :param colored_motifs: a list of motifs to be colored. Other motifs will be colored in grey.
        :param debug: if true, print verbose information.
        """

        methylation_length_total = 0
        upstream_distance = 0
        if ((adjacent_methylation_flag != False) and (adjacent_methylation_flag != "False")):
            if ((repeat_coord_start == 0) or (repeat_coord_end == 0)):
                raise ValueError("Error: Please specify a repeat_coord_start and repeat_coord_end. This region should be the repeat region with the region of interest which will NOT have methylation analyzed on the basis of poor alignment.")
            else:
                methylation_sample_data, methylation_sample_ids, methylation_region_coords = self.generate_adjacent_data(adjacent_methylation_flag, repeat_coord_start, repeat_coord_end)
                methylation_length_total = abs(methylation_region_coords[0][0] - methylation_region_coords[1][1])
                print(methylation_region_coords)

        max_repeat_count = len(aligned_labeled_repeats[0])
        if figure_size is None:
            h = len(sample_ids) / 5 + 2 if len(sample_ids) > 50 else 5
            w = max_repeat_count / 5 + 2 if max_repeat_count > 50 else max_repeat_count // 5 + 2
            if not allele_as_row:
                w, h = h, w
            if h * dpi > 2**16:
                h = 2**15 // dpi * 0.75  # "Weight and Height must be less than 2^16"
            fig, ax_main = plt.subplots(figsize=(w, h))
        else:
            fig, ax_main = plt.subplots(figsize=figure_size)  # width and height in inch

        if title is not None:
            ax_main.set_title(title)

        # Sort by clustering and add dendrogram if needed
        if sort_by_clustering:
            if symbol_to_motif is None:
                raise ValueError("symbol_to_motif must be provided when sort_by_clustering is True")
            sorted_sample_ids, sorted_aligned_labeled_repeats = self.sort_by_clustering(ax_main,
                                                                                        aligned_labeled_repeats,
                                                                                        sample_ids,
                                                                                        symbol_to_motif,
                                                                                        allele_as_row,
                                                                                        hide_dendrogram)
            if debug:
                print("Sort by clustering")
                print('\n'.join(sorted_sample_ids))
        else:
            # Already sorted in some way
            sorted_sample_ids, sorted_aligned_labeled_repeats = sample_ids, aligned_labeled_repeats
            if debug:
                print("No clustering")

        if compression_flag == True:
            sorted_aligned_labeled_repeats = self.compress_motifs(sorted_aligned_labeled_repeats)

        max_repeat_count = len(sorted_aligned_labeled_repeats[0])
        # xaxis_ticks_rounded gives the x-axis position of the first tick for the second region of methylation
        label_positions_final, labels_final, xaxis_ticks_rounded, position_2ndRegion_start, new_start, first_region_length = self.xaxis_tick_label_builder(max_repeat_count, methylation_region_coords)
        
        if figure_size is None:
            h = len(sorted_sample_ids) / 5 + 2 if len(sorted_sample_ids) > 50 else 5
            #w = max_repeat_count / 5 + 2 + (methylation_length_total / 180) if max_repeat_count > 50 else max_repeat_count / 5 + 2 + (methylation_length_total / 180)
            w = max_repeat_count / 5 + 2 + (methylation_length_total / 100) if max_repeat_count > 50 else max_repeat_count / 5 + 2 + (methylation_length_total / 180)
            #w = len(label_positions_final)            
            print(str(w))
            if not allele_as_row:
                w, h = h, w
            if h * dpi > 2**16:
                h = 2**15 // dpi * 0.75  # "Weight and Height must be less than 2^16"
            fig, ax_main = plt.subplots(figsize=(w, h))
            fig.set_size_inches(w, h)

        
        # Set symbol to color map
        self.set_symbol_to_motif_map(aligned_labeled_repeats, alpha, color_palette, colored_motifs, colormap,
                                     symbol_to_motif)

        if ((adjacent_methylation_flag != False) and (adjacent_methylation_flag != "False")):
            upstream_distance = self.draw_methylation_marks(ax_main, box_line_width, no_edge, allele_as_row, sorted_sample_ids, sorted_aligned_labeled_repeats,
                                       methylation_sample_data, methylation_sample_ids, methylation_region_coords, position_2ndRegion_start, new_start,
                                        first_region_length)

        mapped_dict = self.draw_motifs(allele_as_row, ax_main, box_line_width, motif_marks, motif_style, no_edge, private_motif_color,
                         sorted_aligned_labeled_repeats, sorted_sample_ids, xaxis_ticks_rounded)

        # Add another axis for sample labels
        self.add_label_color_axis(aligned_labeled_repeats, allele_as_row, ax_main, box_line_width, sample_to_label,
                                  sorted_aligned_labeled_repeats, sorted_sample_ids, xlabel_rotation, xlabel_size,
                                  ylabel_rotation, ylabel_size)

        self.set_ticks_and_labels(sorted_aligned_labeled_repeats, allele_as_row, ax_main, hide_xticks, hide_yticks,
                                  max_repeat_count, sample_to_label, sorted_aligned_labeled_repeats, sorted_sample_ids,
                                  xlabel, xlabel_rotation, xlabel_size, xtick_offset, xtick_step, ylabel,
                                  ylabel_rotation, ylabel_size, ytick_offset, ytick_step, label_positions_final, labels_final)

        if frame_on is None:  # Set default
            frame_on = {'top': False, 'bottom': True, 'right': False, 'left': True}

        ax_main.spines['top'].set_visible(frame_on['top'])
        ax_main.spines['right'].set_visible(frame_on['right'])
        ax_main.spines['bottom'].set_visible(frame_on['bottom'])
        ax_main.spines['left'].set_visible(frame_on['left'])

        if output_name is not None:
            if '.' not in output_name:
                fig.savefig(f"{output_name}.pdf", dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(f"{output_name}", dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig("test_trplot.png", dpi=dpi, bbox_inches='tight')

        if show_figure:
            plt.show()
        plt.close(fig)
        return mapped_dict

    def xaxis_tick_label_builder(self, max_repeat_count, methylation_region_coords):
        xaxis_ticks_uneven = 0
        label_positions_final = []
        labels_final = []
        space_total = []
        position_2ndRegion_start = 0
        xaxis_ticks_rounded = 0
        first_region_length = 0
        
        xaxis_ticks = 0
        for i in range(methylation_region_coords[0][1] - methylation_region_coords[0][0]):
            xaxis_ticks += 0.02
        xaxis_ticks_rounded = int(-(-xaxis_ticks//1))
        if (xaxis_ticks_rounded % 2) == 0:
            xaxis_ticks_rounded = xaxis_ticks_rounded + 2
        else:
            xaxis_ticks_rounded = xaxis_ticks_rounded + 3
        first_region_length = xaxis_ticks_rounded
        label_positions = [x for x in range(0,xaxis_ticks_rounded,2)]
        labels = [-((xaxis_ticks_rounded*50) - (50*(x+2))) for x in range(0, xaxis_ticks_rounded,2)]
        label_positions_final.extend(label_positions)
        labels_final.extend(labels)

        if (max_repeat_count % 2) == 0:
            label_positions = [(0.5*x)+xaxis_ticks_rounded+0.25 for x in range(0, max_repeat_count)]
            labels = [x for x in range(1, max_repeat_count + 1)]
        else:
            max_repeat_count_new = max_repeat_count + 1
            label_positions = [(0.5*x)+xaxis_ticks_rounded+0.25 for x in range(0, max_repeat_count_new)]
            labels = [x for x in range(1, max_repeat_count + 2)]
        #label_positions_final.extend(label_positions)
        #labels_final.extend(labels)
        position_2ndRegion_start = (label_positions[-1] // 1) + 2

        # re-count second region start
        new_start = ((methylation_region_coords[1][0] - methylation_region_coords[0][0]) // 50) * 50
        xaxis_ticks = 0
        for i in range(methylation_region_coords[1][1] - methylation_region_coords[1][0]):
            xaxis_ticks += 0.02
        xaxis_ticks_rounded2 = int(-(-xaxis_ticks//1))
        if (xaxis_ticks_rounded2 % 2) == 0:
            xaxis_ticks_rounded2 += 2
        else:
            xaxis_ticks_rounded2 += 1
        label_positions = [(x + position_2ndRegion_start) for x in range(0, xaxis_ticks_rounded2 + 1, 2)]
        labels = [(50*x) for x in range(0, xaxis_ticks_rounded2 + 1, 2)]
        label_positions_final.extend(label_positions)
        labels_final.extend(labels)
        
        print(label_positions_final)
        print(labels_final)
        return label_positions_final, labels_final, xaxis_ticks_rounded, position_2ndRegion_start, new_start, first_region_length

    def generate_adjacent_data(self, adjacent_methylation_flag, repeat_coord_start, repeat_coord_end):
        try:
            temporary_placeholder = int(repeat_coord_start)
            temporary_placeholder = int(repeat_coord_end)
        except:
            ValueError("repeat_coord_start or repeat_coord_end is not of the right type (integer). Please use valid arguments and try again.")
        
        f = open(adjacent_methylation_flag, "r")
        filelines = f.readlines()
        f.close()

        methylation_sample_ids = []
        methylation_sample_fps = []
        methylation_sample_chrs = []
        methylation_sample_data = []

        for lines in filelines:
            line = lines.split("\t")
            methylation_sample_ids.append(line[0])
            methylation_sample_fps.append(line[1])
            methylation_sample_chrs.append(line[2])

        ### Establish overall region covered by all bedMethyl files ###
        ### We operate under the assumption that the repeat region is within the bedMethyl region ###
        ### Moreover, it is unclear how we can map methylation within repeats to genomic coordinates (and vice versa) ###
        ### Therefore, we will operate with two regions: upstream and downstream ###
        upstream_start = 250000000
        upstream_end = int(repeat_coord_start)
        downstream_start = int(repeat_coord_end)
        downstream_end = 0
        
        for id, fp, chr in zip(methylation_sample_ids, methylation_sample_fps, methylation_sample_chrs):
            f = open(fp, "r")
            filelines = f.readlines()
            f.close()

            ### We require the chr, the coords and the %mod
            methylation_5mC_lines = []
            methylation_5hmC_lines = []

            for lines in filelines:
                line = lines.split("\t")
                if chr.strip() in line[0]:
                    if int(line[4]) >= 10:
                        if line[3] == "m":
                            methylation_5mC_lines.append(lines)
                        elif line[3] == "h":
                            methylation_5hmC_lines.append(lines)
                    downstream_end = max(int(line[2]), downstream_end)
                    upstream_start = min(int(line[1]), upstream_start)

            methylation_sample_data.append([methylation_5mC_lines, methylation_5hmC_lines])

        return methylation_sample_data, methylation_sample_ids, [[upstream_start, upstream_end], [downstream_start, downstream_end]]

    def compress_motifs(self, sorted_aligned_labeled_repeats):
        sorted_aligned_labeled_repeats_new = []
        for item in sorted_aligned_labeled_repeats:
                sorted_aligned_labeled_repeats_new.append(list(item))
        sorted_aligned_labeled_repeats_df = pd.DataFrame(sorted_aligned_labeled_repeats_new)
        sorted_aligned_labeled_repeats_df = sorted_aligned_labeled_repeats_df.transpose()

        index_list = []
        count_list = []
        symbol_list = []
        last_symbol = "NO_SYMBOL_YET"
        count = 0
        last_index = 0
        total_rows = sorted_aligned_labeled_repeats_df.shape[0] - 1
        index_of_partials = []
        
        for i, row in sorted_aligned_labeled_repeats_df.iterrows():
            
            if last_symbol == "NO_SYMBOL_YET":
                row_checker = []
                count = 0
                j = set(row.to_list())
                set_checker = []
                empty_checker = False
                for curr_j_s in j:
                    if curr_j_s == "-":
                        empty_checker = True
                    else:
                        set_checker.append(curr_j_s)
                if len(set_checker) == 1:
                    last_symbol = set_checker[0]
                    count += 1
                    last_index = i
                    row_checker = row.to_list()
        
            else:
                if row.to_list() == row_checker:
                    count += 1
                else:
                    if count >= 11:
                        index_list.append(last_index)
                        count_list.append(count)
                        symbol_list.append(last_symbol)
                        if empty_checker == True:
                            indices = [str(k) for k, y in enumerate(row_checker) if y == "-"]
                            index_of_partials.append("=".join(indices))
                        else:
                            index_of_partials.append(False)
                    j = set(row.to_list())
                    set_checker = []
                    empty_checker = False
                    for curr_j_s in j:
                        if curr_j_s == "-":
                            empty_checker = True
                        else:
                            set_checker.append(curr_j_s)
                    if len(set_checker) == 1:
                        last_symbol = set_checker[0]
                        count = 1
                        last_index = i
                        row_checker = row.to_list()
                    else:
                        last_symbol = "NO_SYMBOL_YET"
                        count = 0
                        last_index = 0
                        row_checker = []
                        
            if i == total_rows:
                if count >= 11:
                    index_list.append(last_index)
                    count_list.append(count)
                    symbol_list.append(last_symbol)
                    if empty_checker == True:
                        indices = [str(k) for k, y in enumerate(row_checker) if y == "-"]
                        index_of_partials.append("=".join(indices))
                    else:
                        index_of_partials.append(False)
        
                    
        drop_list = []
        for i, c in zip(index_list, count_list):
        	r = range(i+1,i+c)
        	x = [x for x in r]
        	drop_list = drop_list + x
        
        for i,s,c,k in zip(index_list, symbol_list, count_list, index_of_partials):
            if k == False:
                sorted_aligned_labeled_repeats_df.iloc[[i]] = "COMPRESS-" + s + "-" + str(c) + "-NIL"
            else:
                sorted_aligned_labeled_repeats_df.iloc[[i]] = "COMPRESS-" + s + "-" + str(c) + "-" + str(k)
        
        sorted_aligned_labeled_repeats_df.drop(sorted_aligned_labeled_repeats_df.index[drop_list], inplace=True)
        sorted_aligned_labeled_repeats = sorted_aligned_labeled_repeats_df.transpose().to_numpy().tolist()
        print(sorted_aligned_labeled_repeats)
        return sorted_aligned_labeled_repeats
    
    def set_symbol_to_motif_map(self, aligned_labeled_repeats, alpha, color_palette, colored_motifs, colormap,
                                symbol_to_motif):
        unique_labels = self._get_unique_labels(aligned_labeled_repeats)
        unique_label_count = len(unique_labels)
        if colored_motifs is None:
            self.symbol_to_color = self.get_symbol_to_color_map(alpha, unique_label_count, unique_labels,
                                                                color_palette=color_palette,
                                                                colormap=colormap)
        else: # Only assign colors to unique motifs in the colored motifs
            distinct_colors = distinctipy.get_colors(len(colored_motifs), pastel_factor=0.9, rng=777)
            cmap = ListedColormap(distinct_colors)
            if color_palette is not None:
                cmap = plt.get_cmap(color_palette)

            for i, unique_motif in enumerate(colored_motifs):
                for symbol, motif in symbol_to_motif.items():
                    if unique_motif == motif:
                        self.symbol_to_color[symbol] = cmap(i)

            other_motifs = set(unique_labels) - set(self.symbol_to_color.keys())
            for motif in other_motifs:
                self.symbol_to_color[motif] = 'grey'

    def set_ticks_and_labels(self, aligned_labeled_repeats, allele_as_row, ax_main, hide_xticks, hide_yticks,
                             max_repeat_count, sample_to_label, sorted_aligned_labeled_repeats, sorted_sample_ids,
                             xlabel, xlabel_rotation, xlabel_size, xtick_offset, xtick_step, ylabel, ylabel_rotation,
                             ylabel_size, ytick_offset, ytick_step, label_positions_final, labels_final):
        if allele_as_row:
            ax_main.set_ylim(top=len(sorted_aligned_labeled_repeats))
            if sample_to_label is None:
                if ytick_step == 1:
                    ax_main.set_yticks([x + 0.5 for x in range(len(aligned_labeled_repeats))])
                    ax_main.set_yticklabels(sorted_sample_ids, ha='right', rotation=ylabel_rotation)
                else:
                    ax_main.yaxis.set_major_locator(IndexLocator(base=ytick_step, offset=ytick_offset))
            else:
                ax_main.set_yticks([])

            if xtick_step == 1:
                #label_positions = [x + 0.5 for x in range(max_repeat_count)]
                #labels = [x for x in range(1, max_repeat_count + 1)]
                ax_main.set_xticks(label_positions_final, labels=labels_final, rotation=xlabel_rotation)
                ax_main.set_xticklabels(labels_final, rotation=xlabel_rotation)
                print("SUCCESS")
            else:
                ax_main.xaxis.set_major_locator(IndexLocator(base=xtick_step, offset=xtick_offset))
            ax_main.set_xlim(right=max(label_positions_final))
        else:
            ax_main.set_xlim(right=len(sorted_aligned_labeled_repeats))
            if sample_to_label is None:
                if xtick_step == 1:
                    ax_main.set_xticks([x + 0.5 for x in range(len(aligned_labeled_repeats))])
                    ax_main.set_xticklabels(sorted_sample_ids, rotation=xlabel_rotation)
                else:
                    ax_main.xaxis.set_major_locator(IndexLocator(base=xtick_step, offset=xtick_offset))
            else:
                ax_main.set_xticks([])

            if ytick_step == 1:
                label_positions = [y + 0.5 for y in range(max_repeat_count)]
                labels = [y for y in range(1, max_repeat_count + 1)]
                ax_main.set_yticks(label_positions, labels=labels, rotation=ylabel_rotation)
            else:
                ax_main.yaxis.set_major_locator(IndexLocator(base=ytick_step, offset=ytick_offset))
            ax_main.set_ylim(top=max_repeat_count)
        ax_main.tick_params(axis='y', which='major', labelsize=ylabel_size)
        ax_main.tick_params(axis='y', which='minor', labelsize=ylabel_size)
        ax_main.tick_params(axis='x', which='major', labelsize=xlabel_size)
        ax_main.tick_params(axis='x', which='minor', labelsize=xlabel_size)
        if hide_xticks:
            ax_main.set_xticks([])
        if hide_yticks:
            ax_main.set_yticks([])
        # Label
        if xlabel is not None:
            ax_main.set_xlabel(xlabel, fontsize=xlabel_size)
        if ylabel is not None:
            ax_main.set_ylabel(ylabel, fontsize=ylabel_size)

    def draw_methylation_marks(self, ax_main, box_line_width, no_edge, allele_as_row, sorted_sample_ids,
                               sorted_aligned_labeled_repeats, methylation_sample_data, methylation_sample_ids,
                               methylation_region_coords, max_repeat_count_andInitial, new_start, first_region_length):

        ### We operate under the assumption that methylation is consistent across both alleles ###
        ### Map methylation_sample_data to sorted_samples_ids ###

        upstream_distance = 0
        box_height = 1.0
        box_width = 0.02
        for allele_index, (allele, SV_sample_id) in enumerate(zip(sorted_aligned_labeled_repeats, sorted_sample_ids)):
            motif_index = 0
            # each column is an allele
            box_position = [allele_index, 0]
            # use SV_sample_id to reference methylation_sample_ids and find index for methylation_sample_data
            methylation_current_sample_data = methylation_sample_data[methylation_sample_ids.index(SV_sample_id.split(".")[0])]
            
            if allele_as_row:
                box_position = [0, allele_index]

            final_coordinate_index = 0
            for coordinate_index in range(abs(methylation_region_coords[0][0] - methylation_region_coords[0][1])):
                current_coordinate_abs = coordinate_index + methylation_region_coords[0][0]
                final_coordinate_index = first_region_length*50 - abs(methylation_region_coords[0][0] - methylation_region_coords[0][1]) + coordinate_index - 100
                if allele_as_row:
                    box_position[0] = box_width * final_coordinate_index  # move x position
                    upstream_distance = box_width * final_coordinate_index
                else:
                    box_position[1] = box_height * final_coordinate_index
                    upstream_distance = box_height * final_coordinate_index
                special_to_remove = 15
                for coordinate_line in methylation_current_sample_data[0]: # use index 0 which is for 5mC, not 5hmC
                    special_to_remove = special_to_remove - 1
                    coordinate_line = coordinate_line.split("\t")
                    if int(coordinate_line[1]) == int(current_coordinate_abs):
                        #print(coordinate_line)
                        #print(str(coordinate_line[10]))
                        #print(str(float(coordinate_line[10])/100))
                        #print(mpl.colormaps["bwr"]((float(coordinate_line[10])+1)/100))
                        #print(str(1/special_to_remove))
                        fcolor = mpl.colormaps["autumn"](1-( (float(coordinate_line[10])+0.1) / 100 )) # get color from yellow (0) to red (1)
                        ax_main.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                                        linewidth=box_line_width + 0.1,
                                                        facecolor=fcolor,
                                                        edgecolor=fcolor if no_edge else "white",))

            offset_count = 0
            offset_secondRegion = abs(methylation_region_coords[1][1] - methylation_region_coords[0][0]) - new_start - abs(methylation_region_coords[1][0] - methylation_region_coords[1][1])
            for coordinate_index in range(offset_secondRegion + abs(methylation_region_coords[1][0] - methylation_region_coords[1][1])):
                current_coordinate_abs = coordinate_index + methylation_region_coords[1][0] - offset_secondRegion
                if offset_count > offset_secondRegion:
                    if allele_as_row:
                        box_position[0] = (box_width * coordinate_index) + max_repeat_count_andInitial + (offset_secondRegion * box_width)  # move x position
                    else:
                        box_position[1] = (box_height * coordinate_index) + max_repeat_count_andInitial + (offset_secondRegion * box_width)
    
                    for coordinate_line in methylation_current_sample_data[0]: # use index 0 which is for 5mC, not 5hmC
                        coordinate_line = coordinate_line.split("\t")
                        if int(coordinate_line[1]) == int(current_coordinate_abs):
                            fcolor = mpl.colormaps["autumn"](1-( (float(coordinate_line[10])+0.1) / 100 )) # get color from yellow (0) to red (1)
                            ax_main.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                                            linewidth=box_line_width + 0.1,
                                                            facecolor=fcolor,
                                                            edgecolor=fcolor if no_edge else "white",))
                else:
                    offset_count += 1
        #for coordinate_index in range(abs(methylation_region_coords[1][0] - methylation_region_coords[1][1])):
        #    current_coordinate_abs = coordinate_index + methylation_region_coords[1][0]
        #    if allele_as_row:
        #        box_position[0] = (box_width * coordinate_index) + max_repeat_count_andInitial  # move x position
        #    else:
        #        box_position[1] = (box_height * coordinate_index) + max_repeat_count_andInitial
        #    for coordinate_line in methylation_current_sample_data[0]: # use index 0 which is for 5mC, not 5hmC
        #        coordinate_line = coordinate_line.split("\t")
        #        if int(coordinate_line[1]) == int(current_coordinate_abs):
        #            fcolor = mpl.colormaps["autumn"](1-( (float(coordinate_line[10])+0.1) / 100 )) # get color from yellow (0) to red (1)
        #            ax_main.add_patch(plt.Rectangle(box_position, box_width, box_height,
        #                                            linewidth=box_line_width + 0.1,
        #                                            facecolor=fcolor,
        #                                            edgecolor=fcolor if no_edge else "white",))
        return upstream_distance

    def draw_motifs(self, allele_as_row, ax_main, box_line_width, motif_marks, motif_style, no_edge,
                    private_motif_color, sorted_aligned_labeled_repeats, sorted_sample_ids, xaxis_ticks_rounded):

        color_key = []
        symbol_key = []
        compress_tag = False
        box_height = 1.0
        box_width = 0.5
        for allele_index, allele in enumerate(sorted_aligned_labeled_repeats):
            motif_index = 0
            # each column is an allele
            box_position = [allele_index, 0]
            if allele_as_row:
                box_position = [0, allele_index]

            for box_index, symbol in enumerate(allele):
                if allele_as_row:
                    box_position[0] = (box_width * box_index) + xaxis_ticks_rounded  # move x position
                else:
                    box_position[1] = (box_height * box_index) + xaxis_ticks_rounded
                hatch_pattern = None
                if symbol == '-':  # For gaps, color them as white blocks
                    fcolor = (1, 1, 1, 1)
                else:  # Not a gap or private motif
                    if motif_marks is not None and sorted_sample_ids[allele_index] in motif_marks:
                        motif_mark = motif_marks[sorted_sample_ids[allele_index]][motif_index]
                        if motif_mark == 'I':  # introns
                            hatch_pattern = 'xxx'
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    if "COMPRESS" in symbol:
                        if str(allele_index) == symbol.split("-")[3]:
                            fcolor = (1, 1, 1, 1)
                            compress_tag = False
                        else:
                            fcolor = self.symbol_to_color[symbol.split("-")[1]]
                            compress_tag = "<" + str(symbol.split("-")[2]) + ">"
                            if symbol.split("-")[1] not in symbol_key:
                                symbol_key.append(symbol.split("-")[1])
                                color_key.append(fcolor)
                    else:
                        fcolor = self.symbol_to_color[symbol]
                        compress_tag = False
                        if symbol not in symbol_key:
                            symbol_key.append(symbol)
                            color_key.append(fcolor)
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    # fcolor = self.symbol_to_color[symbol]
                    motif_index += 1

                if symbol == PRIVATE_MOTIF_LABEL:
                    fcolor = private_motif_color

                if motif_style == "box":
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    ax_main.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                                    linewidth=box_line_width + 0.1,
                                                    facecolor=fcolor,
                                                    edgecolor=fcolor if no_edge else "white",
                                                    hatch=hatch_pattern, ))
                    if compress_tag != False:
                        ax_main.annotate(compress_tag, (box_position[0] + (box_width / 2), box_position[1] + (box_height / 2)),
                                        color = "w", weight = "bold", fontsize = int(box_height * 3), ha="center", va="center")
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    #ax_main.add_patch(plt.Rectangle(box_position, box_width, box_height,
                    #                                linewidth=box_line_width + 0.1,
                    #                                facecolor=fcolor,
                    #                                edgecolor=fcolor if no_edge else "white",
                    #                                hatch=hatch_pattern, ))
                elif motif_style == "arrow":
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    ax_main.add_patch(plt.Arrow(box_position[0], box_position[1] + box_height / 2, box_width, 0,
                                                    width=1,
                                                    linewidth=box_line_width + 0.1,
                                                    facecolor=fcolor,
                                                    edgecolor=fcolor if no_edge else "white",
                                                    hatch=hatch_pattern, ))
                    if compress_tag != False:
                        ax_main.annotate(compress_tag, (box_position[0] + (box_width / 2), box_position[1] + (box_height / 2)),
                                        color = "w", weight = "bold", fontsize = int(box_height * 3), ha="center", va="center")
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    #ax_main.add_patch(plt.Arrow(box_position[0], box_position[1] + box_height / 2, box_width, 0,
                    #                            width=1,
                    #                            linewidth=box_line_width + 0.1,
                    #                            facecolor=fcolor,
                    #                            edgecolor=fcolor if no_edge else "white",
                    #                            hatch=hatch_pattern, ))

                elif motif_style == "triangle":
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    ax_main.add_patch(plt.Polygon([(box_position[0], box_position[1]),
                                                   (box_position[0], box_position[1] + box_height),
                                                   (box_position[0] + box_width, box_position[1] + box_height / 2)],
                                                  color=fcolor,
                                                  linewidth=box_line_width + 0.1,
                                                 ))
                    if compress_tag != False:
                        ax_main.annotate(compress_tag, (box_position[0] + (box_width / 2), box_position[1] + (box_height / 2)),
                                        color = "w", weight = "bold", fontsize = int(box_height * 3), ha="center", va="center")
                    ###########################################################################
                    # Temporary insertion to handle compressed symbols in format COMPRESS-S-N #
                    ###########################################################################
                    #ax_main.add_patch(plt.Polygon([(box_position[0], box_position[1]),
                    #                               (box_position[0], box_position[1] + box_height),
                    #                               (box_position[0] + box_width, box_position[1] + box_height / 2)],
                    #                              color=fcolor,
                    #                              linewidth=box_line_width + 0.1,
                    #                              ))
                else:
                    raise ValueError(f"Unknown motif style: {motif_style}")
            print("Printing for actual STRs now.")
            print(str(allele_index))
            print(str(box_position[0]))
        mapped_dict = dict(zip(symbol_key, color_key))
        return mapped_dict

    def add_label_color_axis(self, aligned_labeled_repeats, allele_as_row, ax_main, box_line_width, sample_to_label,
                             sorted_aligned_labeled_repeats, sorted_sample_ids, xlabel_rotation, xlabel_size,
                             ylabel_rotation, ylabel_size):
        if sample_to_label is not None:
            box_height = 1.0
            box_width = 1.0
            motif_size = box_width / len(aligned_labeled_repeats[0])
            population_color_map = self.get_sample_to_colormap(sample_to_label)
            if allele_as_row:
                # the half size of one motif.
                ax_left = ax_main.inset_axes([-motif_size/2, 0, motif_size/2, 1])

                for allele_index, allele in enumerate(sorted_aligned_labeled_repeats):
                    sample_id = sorted_sample_ids[allele_index]
                    if sample_id.find('-') != -1:
                        sample_id = sample_id[:sample_id.index('-')]
                    box_position = [0, allele_index]
                    fcolor = population_color_map.get(sample_id, '#dbdbdb')  # grey default
                    ax_left.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                                    linewidth=box_line_width + 0.1,
                                                    facecolor=fcolor,
                                                    edgecolor="black"))

                ax_left.set_ylim(top=len(aligned_labeled_repeats))
                ax_left.set_yticks([x + 0.5 for x in range(len(aligned_labeled_repeats))])
                ax_left.set_yticklabels(sorted_sample_ids, rotation=ylabel_rotation, fontsize=ylabel_size)
                ax_left.set_xticks([])
                ax_left.set_frame_on(False)
                # ax_left.tick_params(axis='y', which='both', length=0)  # if don't want the ticks
            else:
                ax_bottom = ax_main.inset_axes([0, -motif_size/2, 1.0, motif_size/2])
                for allele_index, allele in enumerate(sorted_aligned_labeled_repeats):
                    sample_id = sorted_sample_ids[allele_index]
                    if sample_id.find('-') != -1:
                        sample_id = sample_id[:sample_id.index('-')]
                    box_position = [allele_index, 0]
                    fcolor = population_color_map.get(sample_id, '#dbdbdb')  # grey default
                    ax_bottom.add_patch(plt.Rectangle(box_position, box_width, box_height,
                                                      linewidth=box_line_width + 0.1,
                                                      facecolor=fcolor,
                                                      edgecolor="white"))

                ax_bottom.set_xlim(right=len(aligned_labeled_repeats))
                ax_bottom.set_xticks([x + 0.5 for x in range(len(aligned_labeled_repeats))])
                ax_bottom.set_xticklabels(sorted_sample_ids, rotation=xlabel_rotation, fontsize=xlabel_size)
                ax_bottom.set_yticks([])
                ax_bottom.set_frame_on(False)
                # ax_bottom.tick_params(axis='x', which='both', length=0)  # if don't want the ticks

    def sort_by_clustering(self, ax_main, aligned_labeled_repeats, sample_ids, symbol_to_motif, allele_as_row, hide_clustering):
        """
        Perform hierarchical clustering and sort the samples based on the clustering.

        Parameters
        ----------
        ax_main : matplotlib.axes.Axes object
            The main axes object.
        aligned_labeled_repeats : list of list of str
            List of aligned labeled repeats.
        sample_ids : list of str
            List of sample ids.
        symbol_to_motif : dict of str to str
            Dictionary mapping symbols to motifs.
        hide_clustering : bool
            Whether to hide clustering.

        Returns
        -------
        sorted_sample_ids : list of str
            List of sorted sample ids.
        sorted_aligned_labeled_repeats : list of list of str
            List of sorted aligned labeled repeats.
        """
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform
        from trviz.utils import get_distance_matrix
        from trviz.utils import calculate_cost_with_dist_matrix

        distance_matrix = get_distance_matrix(symbol_to_motif)
        dist_mat = np.zeros([len(aligned_labeled_repeats), len(aligned_labeled_repeats)])
        for i in range(len(aligned_labeled_repeats)):
            for j in range(len(aligned_labeled_repeats)):
                tr_1 = aligned_labeled_repeats[i]
                tr_2 = aligned_labeled_repeats[j]
                dist_mat[i, j] = calculate_cost_with_dist_matrix(tr_1, tr_2, distance_matrix, allow_copy_change=True)
        condensed_dist_mat = squareform(dist_mat)

        Y = sch.linkage(condensed_dist_mat, method='single', optimal_ordering=True)
        if not hide_clustering:
            if allele_as_row:
                ax_dendro = ax_main.inset_axes([1, 0, 0.2, 1])  # right
                Z = sch.dendrogram(Y, orientation='right', get_leaves=True, ax=ax_dendro)
            else:
                ax_dendro = ax_main.inset_axes([0, 1, 1, 0.2])  # top
                Z = sch.dendrogram(Y, orientation='top', get_leaves=True, ax=ax_dendro)
            ax_dendro.set_xticks([])
            ax_dendro.set_yticks([])
            ax_dendro.set_frame_on(False)
        else:
            if allele_as_row:
                Z = sch.dendrogram(Y, orientation='right', get_leaves=True, no_plot=True)
            else:
                Z = sch.dendrogram(Y, orientation='top', get_leaves=True, no_plot=True)

        idx2 = Z['leaves']
        sorted_aligned_labeled_repeats = [aligned_labeled_repeats[i] for i in idx2]
        sorted_sample_ids = [sample_ids[i] for i in idx2]

        return sorted_sample_ids, sorted_aligned_labeled_repeats

    @staticmethod
    def get_symbol_to_color_map(alpha, unique_symbol_count, unique_symbols, color_palette=None, colormap=None):
        """ Get a dictionary mapping symbols to colors """

        if colormap is not None:
            cmap = colormap
        else:
            # Color-blind friendly colors (Okabe-Ito)
            if unique_symbol_count <= 7:
                cmap = ListedColormap([(0.902, 0.624, 0),
                                       (0.337, 0.706, 0.914),
                                       (0, 0.620, 0.451),
                                       (0.941, 0.894, 0.259),
                                       (0, 0.447, 0.698),
                                       (0.835, 0.369, 0),
                                       (0.8, 0.475, 0.655)])
            else:
                cmap = distinctipy.get_colors(unique_symbol_count, pastel_factor=0.9, rng=777)
                cmap = ListedColormap(cmap)

            if color_palette is not None:
                cmap = plt.get_cmap(color_palette)

        symbol_to_color = {r: cmap(i) for i, r in enumerate(unique_symbols)}
        return symbol_to_color

    def set_symbol_to_color_map(self, symbol_to_color):
        self.symbol_to_color = symbol_to_color

    def get_sample_to_colormap(self, sample_to_label: Dict[str, str], default_color='#dbdbdb') -> Dict[str, str]:
        """ Allowed labels are super populations: {AMR, AFR, EAS, SAS, EUR} """

        superpopulation_list_allowed = ('AMR', 'AFR', 'EAS', 'SAS', 'EUR')
        # TODO Allow any labels and assign a distinct color to each distinct label
        # TODO other color map options

        # Course et al. 2020 color map
        # popultation_colormap = {'SAS': (117, 213, 253, 100),  # light blue
        #                         'EAS': (219, 219, 219, 100),  # light grey
        #                         'EUR': (211, 250, 121, 100),  # light green+yellow
        #                         'AMR': (252, 211, 119, 100),  # light orange
        #                         'AFR': (255, 137, 215, 100)}  # light pink

        population_colormap = {'SAS': '#1982c4',  # blue
                               'EAS': '#8ac926',  # green
                               'EUR': '#ffca3a',  # yellow
                               'AMR': '#6a4c93',  # purple
                               'AFR': '#ff595e',  # red
                               }

        sample_to_colormap = {}
        for sample in sample_to_label:
            label = sample_to_label[sample]
            color = population_colormap.get(label, default_color)  # default color is grey
            sample_to_colormap[sample] = color

        return sample_to_colormap
