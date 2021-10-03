{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "outputs": [],
   "source": [
    "import torch"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true,
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "def fp_sampling(points, num):\n",
    "\n",
    "    batch_size = points.shape[0]\n",
    "    D = cdist(points, points)\n",
    "    # By default, takes the first point in the list to be the\n",
    "    # first point in the permutation, but could be random\n",
    "    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)\n",
    "    ds = D[:, 0, :]\n",
    "    for i in range(1, num):\n",
    "        idx = torch.argmax(ds, dim=1)\n",
    "        perm[:, i] = idx\n",
    "\n",
    "        ds = torch.minimum(ds, D[:, idx, :].unique(dim=1).squeeze())\n",
    "\n",
    "    return perm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[[0.8744, 0.3557, 0.5941],\n",
      "         [0.3080, 0.3351, 0.4529],\n",
      "         [0.5825, 0.8798, 0.8043],\n",
      "         ...,\n",
      "         [0.8874, 0.9349, 0.8508],\n",
      "         [0.8919, 0.9590, 0.5587],\n",
      "         [0.3837, 0.6692, 0.7043]],\n",
      "\n",
      "        [[0.3990, 0.8676, 0.8513],\n",
      "         [0.7905, 0.5636, 0.0145],\n",
      "         [0.8293, 0.4851, 0.5209],\n",
      "         ...,\n",
      "         [0.6378, 0.7593, 0.7534],\n",
      "         [0.1741, 0.5873, 0.9898],\n",
      "         [0.4494, 0.3359, 0.4654]],\n",
      "\n",
      "        [[0.9055, 0.0448, 0.1481],\n",
      "         [0.0123, 0.3773, 0.9131],\n",
      "         [0.5003, 0.8100, 0.7676],\n",
      "         ...,\n",
      "         [0.3293, 0.6172, 0.3723],\n",
      "         [0.3804, 0.0854, 0.6134],\n",
      "         [0.8539, 0.2990, 0.6763]],\n",
      "\n",
      "        ...,\n",
      "\n",
      "        [[0.6829, 0.0715, 0.6936],\n",
      "         [0.1994, 0.8265, 0.3660],\n",
      "         [0.6042, 0.8077, 0.7401],\n",
      "         ...,\n",
      "         [0.8588, 0.7758, 0.1899],\n",
      "         [0.5183, 0.1497, 0.4672],\n",
      "         [0.5974, 0.8914, 0.7522]],\n",
      "\n",
      "        [[0.5873, 0.5521, 0.4306],\n",
      "         [0.5051, 0.7528, 0.9099],\n",
      "         [0.6830, 0.2829, 0.0841],\n",
      "         ...,\n",
      "         [0.7594, 0.3621, 0.5503],\n",
      "         [0.1209, 0.3801, 0.0464],\n",
      "         [0.6874, 0.5642, 0.6513]],\n",
      "\n",
      "        [[0.0790, 0.6342, 0.6310],\n",
      "         [0.0844, 0.1534, 0.8641],\n",
      "         [0.0478, 0.8026, 0.2380],\n",
      "         ...,\n",
      "         [0.7880, 0.2265, 0.4440],\n",
      "         [0.9986, 0.8478, 0.5461],\n",
      "         [0.0293, 0.6002, 0.3521]]])\n"
     ]
    }
   ],
   "source": [
    "pc = torch.rand((8, 5671, 3))"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'cdist' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mNameError\u001B[0m                                 Traceback (most recent call last)",
      "\u001B[1;32m~\\AppData\\Local\\Temp/ipykernel_25608/3729001077.py\u001B[0m in \u001B[0;36m<module>\u001B[1;34m\u001B[0m\n\u001B[1;32m----> 1\u001B[1;33m \u001B[0mfp_sampling\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mpc\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;36m2048\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m",
      "\u001B[1;32m~\\AppData\\Local\\Temp/ipykernel_25608/1617422325.py\u001B[0m in \u001B[0;36mfp_sampling\u001B[1;34m(points, num)\u001B[0m\n\u001B[0;32m      2\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m      3\u001B[0m     \u001B[0mbatch_size\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mpoints\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mshape\u001B[0m\u001B[1;33m[\u001B[0m\u001B[1;36m0\u001B[0m\u001B[1;33m]\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m----> 4\u001B[1;33m     \u001B[0mD\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mcdist\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mpoints\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mpoints\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m      5\u001B[0m     \u001B[1;31m# By default, takes the first point in the list to be the\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m      6\u001B[0m     \u001B[1;31m# first point in the permutation, but could be random\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;31mNameError\u001B[0m: name 'cdist' is not defined"
     ]
    }
   ],
   "source": [
    "fp_sampling(pc, 2048)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}