"""_NOTE_:
    This function need to plot the optimized population and PF (if PF exist) , and you can use the following code to get the PF data
    from emoc.utils import LoadPFData
    pf = LoadPFData(pf_path, prob_name)
    
    user can provide the path of the PF data file, if the path is not provided then the function will try to load the PF data from the pf_data folder
    the function will return the PF data as a list of list, the size is P X N,  where P is the number of points and N is the number of objectives
    if the PF data is not available then the function will return []
"""


class Scatter:
    def __init__(self):
        pass

    def scatter_1d(self):
        pass

    def scatter_2d(self):
        pass

    def scatter_3d(self):
        pass

    def scatter_parrallel(self):
        pass


#     def draw_pf(
#             self,
#             algorithm,
#             problem,
#             run = 0,
#             gen = None,
#             point_size = 4 ,
#             marker_type = 'circle',
#             line_width = 1,
#             point_color = 'blue',
#             x_label = 'F1',
#             y_label = 'F2',
#             z_label = 'F3',
#             figure_width = 550,
#             figure_height = 500,
#             tick_font_size = 12,
#             label_font_size = 14,
#             axis_line_color = 'black',
#             background_color = 'white',):

#         if algorithm not in self.algorithm_names:
#             raise ValueError(
#                 f"The named algorithm {algorithm} was not evaluated, "
#                 f"available algorithms are {self.algorithm_names}"
#             )

#         if problem not in self.problem_names:
#             raise ValueError(
#                 f"The named algorithm {problem} was not evaluated, "
#                 f"available algorithms are {self.problem_names}"
#             )

#         if run >= self.n_runs:
#             raise ValueError(
#                 f"There are a total of {self.n_runs}, "
#                 f"but run number {run} is passed."
#             )

#         gen = self.n_gen if gen == None else gen
#         pf = np.array(self.results[(algorithm, problem)][run][gen]["PF"])
#         n_objs = pf.shape[1]
#         data_pf = pd.DataFrame(pf, columns=[f"F{i + 1}" for i in range(n_objs)])

#         if n_objs == 3:
#             self._draw_pf_3d(
#                 data_pf,
#                 point_size = point_size ,
#                 marker_type = marker_type,
#                 line_width = line_width,
#                 point_color = point_color,
#                 x_label = x_label,
#                 y_label = y_label,
#                 z_label = z_label,
#                 figure_width = figure_width,
#                 figure_height = figure_height,
#                 tick_font_size = tick_font_size,
#                 label_font_size = label_font_size,
#                 axis_line_color = axis_line_color,
#                 background_color = background_color,
#             )

#         elif n_objs == 2:
#             self._draw_pf_2d(
#                 data_pf,
#                 point_size = point_size ,
#                 marker_type = marker_type,
#                 line_width = line_width,
#                 point_color = point_color,
#                 x_label = x_label,
#                 y_label = y_label,
#                 figure_width = figure_width,
#                 figure_height = figure_height,
#                 tick_font_size = tick_font_size,
#                 label_font_size = label_font_size,
#                 axis_line_color = axis_line_color,
#             )

#         else:
#             raise NotImplementedError(
#                 "Currently only PF of n_dim = {2,3} can be visualized."
#             )

#     def _draw_pf_3d(
#             self,
#             data,
#             point_size = 4 ,
#             marker_type = 'circle',
#             line_width = 1,
#             point_color = 'blue',
#             x_label = 'F1',
#             y_label = 'F2',
#             z_label = 'F3',
#             figure_width = 550,
#             figure_height = 500,
#             tick_font_size = 12,
#             label_font_size = 14,
#             axis_line_color = 'black',
#             background_color = 'white',
#         ):

#         trace = go.Scatter3d(
#             x = data["F1"],
#             y = data["F2"],
#             z = data["F3"],
#             mode = 'markers',
#             marker = dict(
#                 size = point_size,
#                 symbol = marker_type,
#                 line = dict(
#                     width = line_width,
#                     color = axis_line_color
#                 ),
#                 color = point_color
#             )
#         )

#         layout = go.Layout(
#             title = f'Pareto Front',
#             scene = dict(
#                 xaxis = dict(
#                     title = x_label,
#                     titlefont = dict(size = label_font_size, color = axis_line_color),
#                     tickfont = dict(size = tick_font_size, color = axis_line_color),
#                     showgrid = True,
#                     gridcolor = 'darkgray',
#                     showbackground = False,
#                 ),
#                 yaxis = dict(
#                     title = y_label,
#                     titlefont = dict(size = label_font_size, color = axis_line_color),
#                     tickfont = dict(size = tick_font_size, color = axis_line_color),
#                     showgrid = True,
#                     gridcolor = 'darkgray',
#                     showbackground = False,
#                 ),
#                 zaxis = dict(
#                     title = z_label,
#                     titlefont = dict(size = label_font_size, color = axis_line_color),
#                     tickfont = dict(size = tick_font_size, color = axis_line_color),
#                     showgrid = True,
#                     gridcolor ='darkgray',
#                     showbackground = False,
#                 ),
#                 camera = dict(
#                     eye = dict(x = 1.5, y = 1.5, z = 1.5)
#                 )
#             ),
#             autosize = False,
#             width = figure_width,
#             height = figure_height,
#             margin = dict(l = 50, r = 50, b = 50, t = 50),
#             paper_bgcolor = background_color,
#         )

#         fig = go.Figure(data = [trace], layout = layout)
#         fig.show()

#     def _draw_pf_2d(
#             self,
#             data,
#             point_size = 10,
#             marker_type = 'circle',
#             line_width = 1,
#             point_color = 'blue',
#             x_label = 'F1',
#             y_label = 'F2',
#             figure_width = 550,
#             figure_height = 500,
#             tick_font_size = 16,
#             label_font_size = 20,
#             axis_line_color = 'black',
#             axis_line_width = 1.5,
#             tick_line_width = 1.5
#         ):

#         trace = go.Scatter(
#             x = data["F1"],
#             y = data["F2"],
#             mode = 'markers',
#             marker = dict(
#                 size = point_size,
#                 symbol = marker_type,
#                 line = dict(
#                     width = line_width,
#                     color = axis_line_color
#                 ),
#                 color = point_color
#             )
#         )

#         layout = go.Layout(
#             title = f'Pareto Front',
#             xaxis = dict(
#                 title = x_label,
#                 showline = True,
#                 linecolor = 'black',
#                 linewidth = axis_line_width,
#                 color = 'black',
#                 showgrid = True,
#                 mirror = True,
#                 ticks = 'inside',
#                 tickwidth = tick_line_width,
#                 tickfont = dict(size = tick_font_size),
#                 title_font = dict(size = label_font_size),
#             ),
#             yaxis = dict(
#                 title = y_label,
#                 showline = True,
#                 linecolor = 'black',
#                 linewidth = axis_line_width,
#                 color = 'black',
#                 showgrid = True,
#                 mirror = True,
#                 ticks = 'inside',
#                 tickwidth = tick_line_width,
#                 tickfont = dict(size = tick_font_size),
#                 title_font = dict(size = label_font_size),
#             ),
#             plot_bgcolor = 'white',
#             paper_bgcolor = 'white',
#             width = figure_width,
#             height = figure_height,
#             font = dict(
#                 color = 'black'
#             )
#         )

#         fig = go.Figure(data = [trace], layout = layout)
#         fig.show()
