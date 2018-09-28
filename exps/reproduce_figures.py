from run_experiments import plot_from_res, plot_from_res_gain
"""
Gain experiments
"""
n = 500
k = 10
dim_l = [3, 5, 7, 10, 15]
ampl_l = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
# ampl_l += [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
n_avrg = 100
# type_matrix_l = ['CircularStrongDecrease']
type_matrix_l = ['CircularStrongDecrease', 'LinearStrongDecrease',
                 'CircularBanded', 'LinearBanded']
scaled = True

save_res_dir = '/home/thomas/Dropbox/RobustSeriationEmbeddingBiblio/results_cluster_new'
# Make the gain
for type_matrix in type_matrix_l:
    fig_name = "kendall-tau-vs-noise-for-several-dims-typematrix_{}.pdf" \
               "".format(type_matrix)
    fig_name = save_res_dir + '/' + fig_name

    plot_from_res_gain(n, k, dim_l, ampl_l, type_matrix, scaled,
                       type_lap_l='random_walk', n_avrg=n_avrg,
                       save_res_dir=save_res_dir,
                       save_fig_path=fig_name)

"""
Scaling experiments
"""
n = 500
k = 10
dim_l = [20]
ampl_l = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
n_avrg = 100
# type_matrix_l = ['CircularStrongDecrease']
type_matrix_l = ['CircularStrongDecrease', 'LinearStrongDecrease',
                 'CircularBanded', 'LinearBanded']
scaled_l = [True, 'CTD', False]

save_res_dir = '/home/thomas/Dropbox/RobustSeriationEmbeddingBiblio/results_cluster_new'
for type_matrix in type_matrix_l:
    fig_name = "kendall-tau-vs-noise-for-several-dims-typematrix_{}_scaling.pdf"\
               "".format(type_matrix)
    fig_name = save_res_dir + '/' + fig_name

    plot_from_res(n, k, dim_l[0], ampl_l, type_matrix, scaled_l,
                  type_lap_l='random_walk', n_avrg=n_avrg,
                  save_res_dir=save_res_dir,
                  save_fig_path=fig_name)
