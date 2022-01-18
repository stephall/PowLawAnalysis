# utils.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

# Remark:
# The chapters here are the same as in the jupyter notebook 'Analysis.ipynb', so that
# one can more easily link helper functions (here) and analysis code (jupyter notebook).

####################################################################################
###### 4) DIAGNOSE SAMPLING TRACES
####################################################################################
# Remark: The traces of the chains should be checked to assess if the sampling has worked 
#         and also to determine an appropriate number of burnin steps.
def plot_trace(parameter_name, trace, ax=None, alpha=0.5):
    """"
    Plot the trace (actually the trace of all chains) of the samples.
    
    Args:
        parameter_name (str): Name of the parameter that we want the marginal distribution to
            be approximated by a histogram of its samples.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        ax (axis object): Axis the plot is added to. If None, make a figure and get its axis.
            [Default None]
        alpha (float in [0, 1]): Opacity of the lines.
            [Default 0.5]
        
    Returns:
        None
    """
    # If no axis is passed, make a figure and get its axis
    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()
        show_fig = True
    else:
        show_fig = False
    
    # Get a list of the parameter samples of each chain.
    # Remark: We want to show all samples, so we set keyword 'burn' to 0.
    #         Only samples with an index bigger (or equal) to 'burn' are returned from .get_values(...).
    #         The keyword 'combine' controls if the samples of the chains are combined or not when returned.
    traces_list = trace.get_values(parameter_name, burn=0, combine=False)
        
    # Plot the histogram of the parameter's samples
    for chain_index, chain_trace in enumerate(traces_list):
        plt.plot(chain_trace, color=f"C{chain_index}", alpha=alpha, label=f"Chain {chain_index}")
    ax.set_xlabel('steps', fontsize=20)
    ax.set_ylabel(parameter_name, fontsize=20)
    ax.legend(fontsize=15)
    
    # Show the figure if we made it in this function
    if show_fig:
        plt.show()

####################################################################################
###### 5) VISUALIZE THE POSTERIOR DISTRIBUTION OF THE PARAMETERS
####################################################################################
def plot_marginal_posterior(parameter_name, trace, discard=0, ax=None, color='b', bins=100, alpha=0.5):
    """"
    Plot the marginal posterior distribution of a parameter by approximating it by a histogram of the
    parameter's samples.
    
    Args:
        parameter_name (str): Name of the parameter that we want the marginal distribution to
            be approximated by a histogram of its samples.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
        ax (axis object): Axis the plot is added to. If None, make a figure and get its axis.
            [Default None]
        color (str/rgb-triple): Color for the histogram bars.
            [Default 'b']
        bins (int): Number of bins used for the histogram.
            [Default 100]
        alpha (float in [0, 1]): Opacity of the histogram bars.
            [Default 0.5]
        
    Returns:
        None
    """
    # If no axis is passed, make a figure and get its axis
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        show_fig = True
    else:
        show_fig = False
        
    # Get the parameter samples (combine samples of different chains)
    parameter_samples =  trace.get_values(parameter_name, burn=discard, combine=True)
    
    # Plot the histogram of the parameters's samples
    plt.hist(parameter_samples, bins=bins, density=True, histtype='stepfilled', color=color, alpha=alpha)

    # Set the axis labels
    plt.xlabel(parameter_name, fontsize=20)
    plt.ylabel('Density', fontsize=20)
    
    # Show the figure if we made it in this function
    if show_fig:
        plt.show()
        
def make_scatter_pair_plot(parameter_name_x, parameter_name_y, trace, discard=0, ax=None, color='b', alpha=0.5, marker_size=0.01):
    """"
    Make a pair plot of the samples of a parameter pair.
    
    Args:
        parameter_name_x (str): Name of the parameter that we want to plot along the x axis.
        parameter_name_y (str): Name of the parameter that we want to plot along the y axis.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
        ax (axis object): Axis the plot is added to. If None, make a figure and get its axis.
            [Default None]
        color (str/rgb-triple): Color for the markers.
            [Default 'b']
        alpha (float in [0, 1]): Opacity of the markers.
            [Default 0.5]
        marker_size (float): Size of the markers.
            [Default 0.01]
        
    Returns:
        None
    """
    # If no axis is passed, make a figure and get its axis
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        show_fig = True
    else:
        show_fig = False
        
    # Get the parameter samples (combine samples of different chains)
    parameter_x_samples =  trace.get_values(parameter_name_x, burn=discard, combine=True)
    parameter_y_samples =  trace.get_values(parameter_name_y, burn=discard, combine=True)
    
    # Plot the samples using a scatter plot
    plt.scatter(parameter_x_samples, parameter_y_samples, color=color, alpha=alpha, s=marker_size)

    # Set the axis labels
    plt.xlabel(parameter_name_x, fontsize=20)
    plt.ylabel(parameter_name_y, fontsize=20)
    
    # Show the figure if we made it in this function
    if show_fig:
        plt.show()

def make_contour_pair_plot(parameter_name_x, parameter_name_y, trace, discard=0, ax=None, color='b', alpha=0.5, marker_size=0.01):
    """"
    Make a pair plot of the samples of a parameter pair.
    
    Args:
        parameter_name_x (str): Name of the parameter that we want to plot along the x axis.
        parameter_name_y (str): Name of the parameter that we want to plot along the y axis.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
        ax (axis object): Axis the plot is added to. If None, make a figure and get its axis.
            [Default None]
        color (str/rgb-triple): Color for the markers.
            [Default 'b']
        alpha (float in [0, 1]): Opacity of the markers.
            [Default 0.5]
        marker_size (float): Size of the markers.
            [Default 0.01]
        
    Returns:
        None
    """
    # If no axis is passed, make a figure and get its axis
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        show_fig = True
    else:
        show_fig = False
        
    # Get the parameter samples (combine samples of different chains)
    parameter_x_samples =  trace.get_values(parameter_name_x, burn=discard, combine=True)
    parameter_y_samples =  trace.get_values(parameter_name_y, burn=discard, combine=True)

    
    # Make a contour plot
    make_contour_plot(parameter_x_samples, parameter_y_samples, ax=ax, 
                      contour_colors_dict={0.95: lighter_color(color, 0.25), 0.68: lighter_color(color, 0.5)}, 
                      contour_type='filled', alpha=1, zorder=0)

    # Plot the mean as a cross
    ax.plot([np.mean(parameter_x_samples)], [np.mean(parameter_y_samples)], 'x', color=color, ms=10, label='Mean')

    # Remark: Add fake polygon patches for the legend
    xy      = np.array([[0,0,0,0], [0,0,0,0]]).T
    polygon = mpatches.Polygon(xy, closed=True, color=lighter_color(color, 0.5),  label="68% HPD")
    ax.add_patch(polygon)
    polygon = mpatches.Polygon(xy, closed=True, color=lighter_color(color, 0.25), label="95% HPD")
    ax.add_patch(polygon)

    # Make a legend
    ax.legend(fontsize=15)

    # Set axis limits
    ax.set_xlim([parameter_x_samples.min(), parameter_x_samples.max()])
    ax.set_ylim([parameter_y_samples.min(), parameter_y_samples.max()])

    # Set the axis labels
    ax.set_xlabel(parameter_name_x, fontsize=20)
    ax.set_ylabel(parameter_name_y, fontsize=20)
    
    # Show the figure if we made it in this function
    if show_fig:
        plt.show()

def lighter_color(color, cscale):
    """ Make the input color lighter. """
    # Map the color to rgb triple (array)
    color_rgb = np.array( mcolors.to_rgb(color) )

    # Return the lighter rgb triple
    return cscale*color_rgb + (1-cscale)*np.array([1, 1, 1])

def make_contour_plot(x1, x2, ax=None, bins=250, contour_colors_dict={0.68: 'b'}, contour_type='lines', alpha=1, zorder=0):
    """
    Make a contour plot.

    Args:
        x1 (1d-array): Array containing the samples of the x-axis quantity.
        x2 (1d-array): Array containing the samples of the y-axis quantity.

    Kwargs:
        ax (axis object): Axis the plot is added to. If None, make a figure and get its axis.
            [Default None]
        bins (int): Number of bins used along x and y to 
        contour_colors_dict (dict): Dictionary containing the HPD (in % thus in [0,1[) as keys and 
            their colors as values (str or rgb triple). [Default {0.68: 'b'}]
        contour_type: How should the contours be drawn?
            Options: (1) 'lines': Display the contour boundaries as lines and display the contour value.
                     (2) 'filled': Display the contour as filled area.
            [Default 'lines']
        alpha (float in [0, 1]): Opacity of the contour.
            [Default 1]
        zorder (float/int): Zorder of the contour plot.
            [Default 0]

    Returns:
        None
    """

    # If the axis is not passed, make a figure and get its axis
    make_fig = False
    if ax is None:
        make_fig = True
        plt.figure()
        ax = plt.gca()

    # Get the Histogram 'H' as 2d matrix and the bin edges along x (xe) and y (ye)
    H, xe, ye = np.histogram2d(x1, x2, bins=bins)

    # From 'Help Page' of numpy.histogram2d:
    # Remark: Please note that the histogram does not follow the Cartesian convention where 
    #         x values are on the abscissa and y values on the ordinate axis. Rather, x is 
    #         histogrammed along the first dimension of the array (vertical), and y along the 
    #         second dimension of the array (horizontal). This ensures compatibility with 'histogramdd'.
    # This means that H has entries x along its first (rows) and y along its second axis (columns)
    # so that x and y are unintuitively swapped.
    # Transpose H so that y is constant within one column and changes for different rows 
    # and x is constant within one row and changes for different columns:
    H = H.T

    # Get the sorted list of the fractions in descending order
    sorted_fractions = list(np.sort(np.array(list(contour_colors_dict.keys())))[::-1])

    # Construct lists with the contour thresholds and colors
    contour_colors     = list()
    contour_labels     = list()
    contour_thresholds = list()

    # Loop over the sorted fractions
    for fraction in sorted_fractions:
        # Get the current color
        color = contour_colors_dict[fraction]

        # Append the current color
        contour_colors.append(color)

        # Construct the current contour level as string
        contour_labels.append(f"{int(fraction*100)}%")

        # Determine and append the contour thresholds for the current fraction
        contour_thresholds.append( get_contour_threshold(H, fraction) )

    # In case that contour_type='filled', append the contour threshold for 
    # fraction 0 to the contour thresholds
    if contour_type=='filled':
        contour_thresholds.append( get_contour_threshold(H, 0) )

        # Check that the contour levels are unique
        if len(np.unique(contour_thresholds))<len(contour_thresholds):
            err_msg = f"Contour thresholds must be unique, but they are not!\n" \
                      f"=> Maybe the number of samples is insufficient. Check increasing the number of samples.\n" \
                      f"Remarks:\n" \
                      f"The (sorted) fractions are: {sorted_fractions}\n" \
                      f"The contour thresholds are: {contour_thresholds}\n"
            raise ValueError(err_msg)

    # Smooth the result before plotting
    HH = gaussian_filter(H, 2.99)

    # Differ cases
    if contour_type=='lines':
        # Draw the contours as lines
        c 		 = plt.contour(bin_centers(xe), bin_centers(ye), HH, contour_thresholds, colors=contour_colors, zorder=zorder)
        c.levels = contour_labels
        ax.clabel(c, inline=True, colors=contour_colors)
    elif contour_type=='filled':
        # Draw the contours as filled areas
        c = ax.contourf(bin_centers(xe), bin_centers(ye), HH, contour_thresholds, colors=contour_colors, alpha=alpha, zorder=zorder)
    else:
        # Raise an error
        err_msg = f"The passed value for 'contour_type={contour_type}' is not one of the expected ones 'lines' or 'filled'."
        raise ValueError(err_msg)

def get_contour_threshold(H, target_fraction=0.5):
    """
    Find the counts threshold value so that a certain fraction of pixel counts 
    of the histogram contain counts bigger than the counts threshold value.
    """
    # The histogram H contains the number of samples (counts) within each of the bins.
    # Get the maximal counts corresponding to the maximum of the histogram Matrix H.
    # Remark: The entries of H are floats.
    max_counts = int(H.max())

    # Find the counts threshold value so that a certain fraction of pixel counts 
    # of the histogram contain counts bigger than the counts threshold value.
    # Reduce the count threshold from the maximal counts value until
    # we found the sought for counts threshold (where we break)
    for counts_thresh in range(max_counts, 1, -1):
        # Get the fraction of the sum of counts of pixels with a value bigger
        # than the current counts threshold divided by the sum of all counts
        # of all pixels in the histogram.
        frac = H[H>=counts_thresh].sum() / H.sum()

        # In case this fraction is bigger than the target fraction, break the iteration
        if frac >= target_fraction:
            break

    return counts_thresh

def bin_centers(xe):
    return (xe[1:]+xe[:-1])/2

####################################################################################
###### 6) GET THE PARAMETER ESTIMATES AS SUMMARY STATISTICS
####################################################################################
def get_HDD_bounds(samples, HDD):
    """"
    Return the Highest Distribution Density (HDD) boundaries for input samples.
    
    Args:
        samples (1d-array): Samples of which we would like to determine the HDD boundaries from.
        HDD (int/float): Highest Distribution Density (HDD) [in percentage] which must be in ]0, 100[.
        
    Returns:
        A two element array containting the HDD boundaries.
    """
    # Ensure that samples is an array
    samples = np.array(samples)
    
    # Check that samples is 1d
    if samples.ndim!=1:
        err_msg = f"The input 'samples' must be castable to a 1d array, got dimension {samples.ndim} after cast."
        raise ValueError(err_msg)
    
    # Check that HDD is inside ]0, 100[
    if not (0<HDD and HDD<100):
        err_msg = f"The input 'HDD' must be within ]0, 100[, got a value of {HDD} instead."
        raise ValueError(err_msg)
    
    # Define the boundaries of the lower percentile
    low_percentile_bounds = [0, 100-HDD]
    
    # Set the initial lower percentile in the middle of the boundaries.
    # This way we ensure, that x starts within its boundaries
    low_percentile_init = float(np.mean(low_percentile_bounds))
    
    # Define the const function only dependent on x (=low_percentile) but as scipy.opt.minimize
    # will pass arrays as inputs, define it for a vectorial x (but with only one component).
    # Remark: We use 'HDD' as the inpit 'diff_percentile' as it is the difference between the
    #         low and the upper percentile.
    cost_func_x = lambda x: np.diff(get_percentile_bounds(samples, x[0], HDD))
    
    # Use this cost function to define a cost function ensuring that x (=low_percentile) is
    # within its boundaries.
    # As x will be cast to an array, so will be the boundaries. As low_percentile_bounds is
    # a 2-element list, we make it a list of lists so that it is cast to an array of shape
    # (1, 2) within the function 'cost_func_bounds'.
    cost_func_bounds_x = lambda x: cost_func_bounds(x, cost_func_x, [low_percentile_bounds])
    
    # Optimize the cost function with boundaries using 'Nelder-Mead' as optimization method
    soln = opt.minimize(cost_func_bounds_x, low_percentile_init, method='Nelder-Mead')
    
    # Get the optimal lower percentile
    # Remark: As soln.x is the output of scipy.opt.minimize, it is an array. In our case this
    #         array has only one component (namely low_percentile). Only get this component.
    low_percentile_opt = soln.x[0]

    # Get the percentile boundaries for this optimal lower percentile which 
    # corresponds to the sought for HDD boundaries
    HDD_bounds = get_percentile_bounds(samples, low_percentile_opt, HDD)

    return HDD_bounds

# Define a local percentile bounds function
def get_percentile_bounds(samples, low_percentile, diff_percentile):
    """
    Return the boundary values defined by the lower percentile and 
    the difference to the upper percentile for some samples.
    """
    # Construct the upper percentile from the lower percentile and 
    # the percentile difference
    upp_percentile = low_percentile + diff_percentile

    # Return the percentile boundaries
    return np.percentile(samples, [low_percentile, upp_percentile])

# Define a const function with boundaries
def cost_func_bounds(x, cost_func_x, x_bounds):
    """"
    Define a cost function with boundaries.

    Args:
        x (castable to 1d array): Input to the cost function, which be castable to
            a 1d numpy array and will have shape (#components,) after the cast.
            E.g. a number or a list/tuple of numbers.
        cost_func_x (function): A cost function which returns a number for the input x.
        x_bounds (castable to 2d array): Boundaries of the components of x, which must
            be castable to a 2d array of shape (#components, 2) after the cast.
            E.g. a list/tuple of 2-element lists/tuples.
    Return:
        If all components of x are within their boundaries, return the cost_func_x(x) value.
        Otherwise, if one component is outside of its boundaries, return inf.
    """
    # Cast x and x_bounds to arrays
    x_arr        = np.array(x)
    x_bounds_arr = np.array(x_bounds)

    # Check that x_arr is 1d
    if x_arr.ndim!=1:
        err_msg = f"The input 'x' must be castable to a 1d numpy array but got dimension {x_arr.ndim} after cast."
        raise ValueError(err_msg)
    
    # Check that x_bounds is 2d
    if x_bounds_arr.ndim!=2:
        err_msg = f"The input 'x_bounds' must be castable to a 2d numpy array but got dimension {x_bounds_arr.ndim} after cast."
        raise ValueError(err_msg)
        
    # Check that the first axis of both x_arr and x_bounds_arr have the same length
    if x_arr.shape[0]!=x_bounds_arr.shape[0]:
        err_msg = f"The input 'x' and 'x_bounds' must have the same number of elements along their first axes after casting " \
                  f"them to arrays.\nHowever after casting, x has shape {x_arr.shape} and x_bounds has shape {x_bounds_arr.shape}."
        raise ValueError(err_msg)
    
    # Check that the second axis of x_bounds_arr has length two
    if x_bounds_arr.shape[1]!=2:
        err_msg = f"The input 'x_bounds' must have a second axis of length 2 after casting it to an array. However, shape of x_bounds is {x_bounds_arr.shape}."
        raise ValueError(err_msg)
    
    # Return np.inf in case that any component of x_arr is outside its boundaries
    if np.any(x_arr<=x_bounds_arr[:, 0]) or np.any(x_bounds_arr[:, 1]<=x_arr):
        return np.inf

    # Evaluate the cost function
    return cost_func_x(x)

def display_param_estimate(parameter_name, trace, discard=0, HPD=68):
    """"
    Display the parameter estimate of a parameter as mean and HPD interval boundaries
    that we can use for a unimodal marginal distribution as Credible Interval (CI).
    
    Args:
        parameter_name (str): Name of the parameter that we want the marginal distribution to
            be approximated by a histogram of its samples.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
        HPD (int/float): Highest Posterior Density (HPD) [in percentage] which must be in ]0, 100[.
            [Default 68]
        
    Returns:
        None
    """        
    # Get the parameter samples (combine samples of different chains)
    parameter_samples =  trace.get_values(parameter_name, burn=discard, combine=True)
    
    # Determine the mean of the samples
    parameter_mean = np.mean(parameter_samples)
    
    # Determine the HPD of the samples using the function 'get_HDD_bounds' that
    # returns a 2-element list containing the lower and upper HDD boundaries for
    # the input samples.
    parameter_HPD_bounds = get_HDD_bounds(parameter_samples, HPD)
    
    # Print out the estimate in the form "mean+(HPD_bounds_upp-mean)-(mean-HPD_bounds_low)"
    print(f"{parameter_name} = {parameter_mean:.3f}+{parameter_HPD_bounds[1]-parameter_mean:.3f}-{parameter_mean-parameter_HPD_bounds[0]:.3f}")

####################################################################################
###### 7) CONSTRUCT THE POSTERIOR DISTRIBUTION OF THE POWER LAW
####################################################################################
def get_pow_law_samples(T, trace, discard=0):
    """"
    Generate the power law samples samples for a given input of temperatures 'T'.
    
    Args:
        T (1d-array): Values of the temperatures at which we want to obtain the power law samples at.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
    
    Returns:
        The samples matrix with shape (#samples, #T)
    """
    # Get the parameter samples
    T_c_samples   = trace.get_values('T_c', burn=discard, combine=True)
    alpha_samples = trace.get_values('alpha', burn=discard, combine=True)
    beta_samples  = trace.get_values('beta', burn=discard, combine=True)

    # Loop over the parameter samples and determine the power law
    y_samples_list = list()
    for T_c, alpha, beta in zip(T_c_samples, alpha_samples, beta_samples):
        # y is zero for all T above T_c thus initialize y as zeros array
        y = np.zeros_like(T)

        # Assign the power law to all T below T_c
        inds    = np.where(T<=T_c)
        y[inds] = alpha*(T_c-T[inds])**beta
        
        # Append y to y_samples_list
        y_samples_list.append(y)
        
    # Stack the y_samples_list to obtain a 2d array of shape (#samples, #T)
    y_samples = np.vstack(y_samples_list)
    
    return y_samples


def get_HDD_bounds_vectorial(samples_mat, HDD):
    """"
    Return the Highest Distribution Density (HDD) boundaries for input samples.
    
    Args:
        samples_mat (2d-array): Samples matrix of shape (#samples, #points) containing the samples
            evaluated at certain points (e.g. temperature values).
        HDD (int/float): Highest Distribution Density (HDD) [in percentage] which must be in ]0, 100[.
        
    Returns:
        A 2-element list containing 1d arrays with the lower and upper HDD boundaries for each evaluation point.
    """
    # Initialize two lists which will hold the lower and upper HDD values for each point
    HDD_bounds_low_list = list()
    HDD_bounds_upp_list = list()
    
    # Loop over the different points (we have to transpose samples_mat to loop over these)
    for samples in samples_mat.T:
        # Determine the HDD boundaries for the samples of the current point
        HDD_bounds = get_HDD_bounds(samples, HDD=HDD)
        
        # Append these boundaries to their corresponding lists
        # Remark: HDD_bounds is a 2-element list/array containing the lower and upper HDD for
        #         the samples passed to 'get_HDD_bounds'.
        HDD_bounds_low_list.append(HDD_bounds[0])
        HDD_bounds_upp_list.append(HDD_bounds[1])
        
    # Make the lists arrays and return them as 2-element list
    return [np.array(HDD_bounds_low_list), np.array(HDD_bounds_upp_list)]


####################################################################################
###### 8) CONSTRUCT THE POSTERIOR DISTRIBUTION OF THE POWER LAW (INVERTED)
####################################################################################
def get_inverted_pow_law_samples(y, trace, T_max, discard=0):
    """"
    Generate the temperature samples for a given input of y values (power law quantity).
    
    Args:
        y (1d-array): Values of the power law quantity 'y' at which we want to obtain the temperature samples at.
        trace (trace object): Trace object returned from pm.sample() that contains the parameter samples.
        T_max (float/int): Maximal temperature value of the data. This is only used to prepare the arrays
            for the plots.
    
    Kwargs:
        discard (int): The number of steps to be discared (as tuning+burnin steps), when accessing the parameter samples.
            [Default 0]
        
    Returns:
        2-tuple [(y, T_samples)] containing the updated y and the sample matrix 'T_samples' with shape (#samples, #T)
    """
    # Get the parameter samples
    T_c_samples   = trace.get_values('T_c', burn=discard, combine=True)
    alpha_samples = trace.get_values('alpha', burn=discard, combine=True)
    beta_samples  = trace.get_values('beta', burn=discard, combine=True)
    
    # Calculate the y_max[=y(T=0)] samples
    y_max_samples = alpha_samples*T_c_samples**beta_samples

    # Loop over the parameter samples and determine the power law
    T_samples_list = list()
    for T_c, alpha, beta, y_max in zip(T_c_samples, alpha_samples, beta_samples, y_max_samples):
        # T is zero for all y above y_max thus initialize T as zeros array
        T = np.zeros_like(y)

        # Assign the power law to all y below y_max
        inds    = np.where(y<=y_max)
        T[inds] = T_c-y[inds]**(1/beta)/alpha
        
        # The inverted power law is bounded to values lower than T_c, while taking the value 0 above.
        # Thus we do a trick and add the point (T_max, y=0). This means we have add T_max at the start 
        # of the current T sample array
        T = np.array([T_max]+list(T))
        
        # Append T to T_samples_list
        T_samples_list.append(T)
        
    # Stack the y_samples_list to obtain a 2d array of shape (#samples, #T)
    T_samples = np.vstack(T_samples_list)
    
    # As we have added the value T_max to the end of each T sample array, we need now to add y=0
    # at the start of the y array
    y = np.array([0]+list(y))
    
    return y, T_samples
