import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np

percentage=True
plot_type='192'
output_path=os.path.join('/Users/burcakotlu/Desktop/')
project='test_tsb'

if plot_type == '192' or plot_type == '96SB' or plot_type == '384':
		with open(matrix_path) as f:
			next(f)
			first_line = f.readline()
			first_line = first_line.strip().split()
			if first_line[0][6] == ">" or first_line[0][3] == ">":
				pcawg = True
			if first_line[0][7] != "]" and first_line[0][6] != ">" and first_line[0][3] != ">":
				sys.exit("The matrix does not match the correct SBS192 format. Please check you formatting and rerun this plotting function.")
		pp = PdfPages(output_path + 'SBS_384_plots_' + project + '.pdf')
		mutations = OrderedDict()
		try:
			with open (matrix_path) as f:
				first_line = f.readline()
				if pcawg:
					samples = first_line.strip().split(",")
					samples = samples[3:]
					samples = [x.replace('"','') for x in samples]
				else:
					samples = first_line.strip().split("\t")
					samples = samples[1:]

				for sample in samples:
					mutations[sample] = OrderedDict()
					mutations[sample]['C>A'] = OrderedDict()
					mutations[sample]['C>G'] = OrderedDict()
					mutations[sample]['C>T'] = OrderedDict()
					mutations[sample]['T>A'] = OrderedDict()
					mutations[sample]['T>C'] = OrderedDict()
					mutations[sample]['T>G'] = OrderedDict()

				for lines in f:
					if pcawg:
						line = lines.strip().split(",")
						line = [x.replace('"','') for x in line]
						nuc = line[2][0] + "[" + line[1] + "]" + line[2][2]
						bias = line[0][0]
					else:
						line = lines.strip().split()
						nuc = line[0][2:]
						bias = line[0][0]
					if bias == 'N' or bias == 'B':
						continue
					else:
						if pcawg:
							mut_type = line[1]
							sample_index = 3
						else:
							mut_type = line[0][4:7]
							sample_index = 1

						for sample in samples:
							if percentage:
								mutCount = float(line[sample_index])
								if mutCount < 1 and mutCount > 0:
									sig_probs = True
							else:
								try:
									mutCount = int(line[sample_index])
								except:
									print("It appears that the provided matrix does not contain mutation counts.\n\tIf you have provided a signature activity matrix, please change the percentage parameter to True.\n\tOtherwise, ", end='')

								# mutCount = int(line[sample_index])
							if nuc not in mutations[sample][mut_type].keys():
								mutations[sample][mut_type][nuc] = [0,0]
							if bias == 'T':
								mutations[sample][mut_type][nuc][0] = mutCount
							else:
								mutations[sample][mut_type][nuc][1] = mutCount
							sample_index += 1

			sample_count = 0
			for sample in mutations.keys():
				total_count = sum(sum(sum(tsb) for tsb in nuc.values()) for nuc in mutations[sample].values())
				plt.rcParams['axes.linewidth'] = 2
				plot1 = plt.figure(figsize=(43.93,9.92))
				plt.rc('axes', edgecolor='lightgray')
				panel1 = plt.axes([0.04, 0.09, 0.95, 0.77])
				xlabels = []

				x = 0.7
				ymax = 0
				colors = [[3/256,189/256,239/256],
                          [1/256,1/256,1/256],
                          [228/256,41/256,38/256],
                          [203/256,202/256,202/256],
                          [162/256,207/256,99/256],
                          [236/256,199/256,197/256]]
				i = 0
				for key in mutations[sample]:
					for seq in mutations[sample][key]:
						xlabels.append(seq[0]+seq[2]+seq[6])
						if percentage:
							if total_count > 0:
								trans = plt.bar(x, mutations[sample][key][seq][0]/total_count*100,width=0.75,color=[1/256,70/256,102/256],align='center', zorder=1000, label='Transcribed Strand')
								x += 0.75
								untrans = plt.bar(x, mutations[sample][key][seq][1]/total_count*100,width=0.75,color=[228/256,41/256,38/256],align='center', zorder=1000, label='Untranscribed Strand')
								x += .2475
								if mutations[sample][key][seq][0]/total_count*100 > ymax:
										ymax = mutations[sample][key][seq][0]/total_count*100
								if mutations[sample][key][seq][1]/total_count*100 > ymax:
										ymax = mutations[sample][key][seq][1]/total_count*100

						else:
							trans = plt.bar(x, mutations[sample][key][seq][0],width=0.75,color=[1/256,70/256,102/256],align='center', zorder=1000, label='Transcribed Strand')
							x += 0.75
							untrans = plt.bar(x, mutations[sample][key][seq][1],width=0.75,color=[228/256,41/256,38/256],align='center', zorder=1000, label='Untranscribed Strand')
							x += .2475
							if mutations[sample][key][seq][0] > ymax:
									ymax = mutations[sample][key][seq][0]
							if mutations[sample][key][seq][1] > ymax:
									ymax = mutations[sample][key][seq][1]
						x += 1
					i += 1

				x = .0415
				y3 = .87
				y = int(ymax*1.25)
				x_plot = 0

				yText = y3 + .06
				plt.text(.1, yText, 'C>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
				plt.text(.255, yText, 'C>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
				plt.text(.415, yText, 'C>T', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
				plt.text(.575, yText, 'T>A', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
				plt.text(.735, yText, 'T>C', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)
				plt.text(.89, yText, 'T>G', fontsize=55, fontweight='bold', fontname='Arial', transform=plt.gcf().transFigure)

				if y <= 4:
					y += 4

				while y%4 != 0:
					y += 1

				# ytick_offest = int(y/4)
				y = ymax/1.025

				ytick_offest = float(y/3)
				for i in range(0, 6, 1):
					panel1.add_patch(plt.Rectangle((x,y3), .155, .05, facecolor=colors[i], clip_on=False, transform=plt.gcf().transFigure))
					panel1.add_patch(plt.Rectangle((x_plot,0), 32, round(ytick_offest*4, 1), facecolor=colors[i], zorder=0, alpha = 0.25, edgecolor='grey'))
					x += .1585
					x_plot += 32

				if percentage:
					ylabs = [0, round(ytick_offest, 1), round(ytick_offest*2, 1), round(ytick_offest*3, 1), round(ytick_offest*4, 1)]
					ylabels= [str(0), str(round(ytick_offest, 1)) + "%", str(round(ytick_offest*2, 1)) + "%",
							  str(round(ytick_offest*3, 1)) + "%", str(round(ytick_offest*4, 1)) + "%"]
				else:
					ylabs = [0, ytick_offest, ytick_offest*2, ytick_offest*3, ytick_offest*4]
					ylabels= [0, ytick_offest, ytick_offest*2,
							  ytick_offest*3, ytick_offest*4]

				labs = np.arange(0.750,192.750,1)

				font_label_size = 30
				if not percentage:
					if int(ylabels[3]) >= 1000:
						font_label_size = 20

				if percentage:
					if len(ylabels) > 2:
						font_label_size = 20

				if not percentage:
					ylabels = ['{:,}'.format(int(x)) for x in ylabels]
					if len(ylabels[-1]) > 3:
						ylabels_temp = []
						if len(ylabels[-1]) > 7:
							for label in ylabels:
								if len(label) > 7:
									ylabels_temp.append(label[0:-8] + "m")
								elif len(label) > 3:
									ylabels_temp.append(label[0:-4] + "k")
								else:
									ylabels_temp.append(label)

						else:
							for label in ylabels:
								if len(label) > 3:
									ylabels_temp.append(label[0:-4] + "k")
								else:
									ylabels_temp.append(label)
						ylabels = ylabels_temp

				panel1.set_xlim([0, 96])
				panel1.set_ylim([0, y])
				panel1.set_xticks(labs)
				panel1.set_yticks(ylabs)
				count = 0
				m = 0
				for i in range (0, 96, 1):
					plt.text(i/101 + .0415, .02, xlabels[i][0], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
					plt.text(i/101 + .0415, .044, xlabels[i][1], fontsize=30, color=colors[m], rotation='vertical', verticalalignment='center', fontname='Courier New', fontweight='bold',transform=plt.gcf().transFigure)
					plt.text(i/101 + .0415, .071, xlabels[i][2], fontsize=30, color='gray', rotation='vertical', verticalalignment='center', fontname='Courier New', transform=plt.gcf().transFigure)
					count += 1
					if count == 16:
						count = 0
						m += 1

				if sig_probs:
					plt.text(0.045, 0.75, sample, fontsize=60, weight='bold', color='black', fontname= "Arial", transform=plt.gcf().transFigure)
				else:
					plt.text(0.045, 0.75, sample + ": " + "{:,}".format(int(total_count)) + " transcribed subs", fontsize=60, weight='bold', color='black', fontname= "Arial", transform=plt.gcf().transFigure)

				custom_text_upper_plot = ''
				try:
					custom_text_upper[sample_count]
				except:
					custom_text_upper = False
				try:
					custom_text_bottom[sample_count]
				except:
					custom_text_bottom = False

				if custom_text_upper:
					plot_custom_text = True
					if len(custom_text_upper[sample_count]) > 40:
						print("To add a custom text, please limit the string to <40 characters including spaces.")
						plot_custom_text = False
				if custom_text_bottom:
					if len(custom_text_bottom[sample_count]) > 40:
						print("To add a custom text, please limit the string to <40 characters including spaces.")
						plot_custom_text = False

				if plot_custom_text:
					x_pos_custom = 0.84
					if custom_text_upper and custom_text_bottom:
						custom_text_upper_plot = custom_text_upper[sample_count] + "\n" + custom_text_bottom[sample_count]

					if custom_text_upper and not custom_text_bottom:
						custom_text_upper_plot = custom_text_upper[sample_count]
						panel1.text(x_pos_custom, 0.78, custom_text_upper_plot, fontsize=40, weight='bold', color='black', fontname= "Arial", transform=plt.gcf().transFigure, ha='right')

					elif custom_text_upper and custom_text_bottom:
						panel1.text(x_pos_custom, 0.72, custom_text_upper_plot, fontsize=40, weight='bold', color='black', fontname= "Arial", transform=plt.gcf().transFigure, ha='right')

					elif not custom_text_upper and custom_text_bottom:
						custom_text_upper_plot = custom_text_bottom[sample_count]
						panel1.text(x_pos_custom, 0.78, custom_text_upper_plot, fontsize=40, weight='bold', color='black', fontname= "Arial", transform=plt.gcf().transFigure, ha='right')


				panel1.set_yticklabels(ylabels, fontsize=font_label_size)
				plt.gca().yaxis.grid(True)
				plt.gca().grid(which='major', axis='y', color=[0.6,0.6,0.6], zorder=1)
				panel1.set_xlabel('')
				panel1.set_ylabel('')
				plt.legend(handles=[trans, untrans], prop={'size':30})
				if percentage:
					plt.ylabel("Percentage of Single Base Substitutions", fontsize=35, fontname="Times New Roman", weight = 'bold')
				else:
					plt.ylabel("Number of Single Base Substitutions", fontsize=35, fontname="Times New Roman", weight = 'bold')

				panel1.tick_params(axis='both',which='both',\
								   bottom=False, labelbottom=False,\
								   left=False, labelleft=True,\
								   right=False, labelright=False,\
								   top=False, labeltop=False,\
								   direction='in', length=25, colors=[0.6, 0.6, 0.6])

				[i.set_color("black") for i in plt.gca().get_yticklabels()]

				pp.savefig(plot1)
				plt.close()
				sample_count += 1
			pp.close()

		except:
			print("There may be an issue with the formatting of your matrix file.")
			os.remove(output_path + 'SBS_384_plots_' + project + '.pdf')
