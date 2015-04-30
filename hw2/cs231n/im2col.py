import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, verbose=0):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  if verbose > 1: print "field_height: {}, field_width: {}, padding: {}, stride ".format(field_height, field_width, padding, stride)

  i0 = np.repeat(np.arange(field_height), field_width)

  if verbose > 1: print "i0 - step 1: {}".format(str(i0))

  i0 = np.tile(i0, C)
  
  if verbose > 1: print "i0 - step 2: {}".format(str(i0))

  i1 = stride * np.repeat(np.arange(out_height), out_width)

  if verbose > 1: print "i1: {}".format(str(i1))

  j0 = np.tile(np.arange(field_width), field_height * C)

  if verbose > 1: print "j0: {}".format(str(j0))

  j1 = stride * np.tile(np.arange(out_width), out_height)

  if verbose > 1: print "j1: {}".format(str(j1))

  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)


  if verbose > 0: 
    print k.shape
    print i.shape
    print j.shape
    print "(i,j,k):"
    print "\n".join([ str( [ zip(i[x,:].tolist(), j[x,:].tolist()), k[x,:].tolist()] ) for x in xrange(i.shape[0])])

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1, verbose=False):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride, verbose)

  cols = x_padded[:, k, i, j]

  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1, verbose=False):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)

  if verbose:
    print "columns reshaped: {}".format(str(cols_reshaped.shape))
    print "x_padded shape: {}".format(str(x_padded.shape))
    print "k shape: {}\ni shape: {}\nj shape: {}".format(str(k.shape), str(i.shape), str(j.shape))

  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

pass
