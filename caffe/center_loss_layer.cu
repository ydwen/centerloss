#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {
template <typename Dtype>
__global__ void CL_count_gpu(int nthreads, const int M, const Dtype* label, Dtype* count) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    count[index] = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count[index]++;
      }
    }
  }
}

template <typename Dtype>
__global__ void CL_difference_gpu(int nthreads, const int K, const Dtype* bottom,
        const Dtype* label, const Dtype* center, Dtype* difference) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    difference[index] = bottom[index] - center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void CL_L2_propagate_gpu(int nthreads, int K, const Dtype margin, const Dtype* difference,
        Dtype* propagate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype d_norm = (Dtype)0.;
    for (int k = 0; k < K; k++) {
      d_norm += difference[index * K + k] * difference[index * K + k];
    }
    d_norm = sqrt(d_norm);
    propagate[index] = (d_norm - margin) / d_norm;
    propagate[index] = propagate[index] > (Dtype)0. ? propagate[index] : (Dtype)0.;
  }
}

template <typename Dtype>
__global__ void CL_cos_propagate_gpu(int nthreads, int K, const Dtype eps, const Dtype margin, const Dtype* bottom,
        const Dtype* label, const Dtype* center, Dtype* propagate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // g, xx, xc, cc
    Dtype g = margin;
    Dtype r = (Dtype)0.;
    Dtype s = (Dtype)0.;
    Dtype t = (Dtype)0.;
    const int label_value = static_cast<int>(label[index]);
    for (int k = 0; k < K; k++) {
      r += bottom[index * K + k] * bottom[index * K + k];
      s += bottom[index * K + k] * center[label_value * K + k];
      t += center[label_value * K + k] * center[label_value * K + k];
    }
    if (!(s / sqrt(r) / sqrt(t) < g)) {
      propagate[index] = (Dtype)0.;
    } else {
      // a, b, c
      Dtype c = s * s - g * g * r * t;
      Dtype b = (Dtype)2. * (((Dtype)1. - g * g) * s * t - c);
      Dtype a = ((Dtype)1. - g * g) * t * (t - (Dtype)2. * s) + c;
      // x1, x2, propagate
      Dtype delta = b * b - (Dtype)4. * a * c;
      Dtype sqrt_delta = delta > (Dtype)0. ? sqrt(delta) : (Dtype)0.;
      Dtype x1 = (-b + sqrt_delta) / ((Dtype)2. * a);
      Dtype x2 = (-b - sqrt_delta) / ((Dtype)2. * a);
      if (x1*x2 < (Dtype)0. || x1+x2 > (Dtype)1.) {
        propagate[index] = (x1 > (Dtype)0. && x1 < (Dtype)1.000001) ? x1 : x2;
      } else {
        propagate[index] = x1 > x2 ? x1 : x2;
      }
    }
  }
}

template <typename Dtype>
__global__ void CL_propagated_difference_gpu(int nthreads, const int K, const Dtype* propagate,
        Dtype* difference) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    difference[index] = difference[index] * propagate[m];
  }
}

template <typename Dtype>
__global__ void CL_center_diff_gpu(int nthreads, const int K, const Dtype* label,
        const Dtype* count, const Dtype* difference, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    caffe_gpu_atomic_add(-difference[index] / count[label_value], center_diff + label_value * K + k);
  }
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = N_;
  CL_count_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, bottom[1]->gpu_data(), count_.mutable_gpu_data());
  // difference
  nthreads = M_ * K_;
  CL_difference_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(), difference_.mutable_gpu_data());
  // propagate
  nthreads = M_;
  const Dtype eps_ = this->layer_param_.center_loss_param().eps();
  const Dtype margin_ = this->layer_param_.center_loss_param().margin();
  if (this->layer_param_.center_loss_param().type()==CenterLossParameter_DistanceType_L2) {
    CL_L2_propagate_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
       CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, margin_, difference_.gpu_data(),
                                 propagate_.mutable_gpu_data());
  } else if (this->layer_param_.center_loss_param().type()==CenterLossParameter_DistanceType_COSINE) {
    CL_cos_propagate_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
       CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, eps_, margin_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                 this->blobs_[0]->gpu_data(), propagate_.mutable_gpu_data());
  }
  // propagated difference
  nthreads = M_ * K_;
  CL_propagated_difference_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
     CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, propagate_.gpu_data(), difference_.mutable_gpu_data());

  Dtype mean_propagate;
  caffe_gpu_asum(M_, propagate_.gpu_data(), &mean_propagate);
  Dtype dot;
  caffe_gpu_dot(M_ * K_, difference_.gpu_data(), difference_.gpu_data(), &dot);
  top[0]->mutable_cpu_data()[0] = dot / M_ / (Dtype)2.;
  top[1]->mutable_cpu_data()[0] = mean_propagate;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    int nthreads = M_ * K_;
    CL_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[1]->gpu_data(), count_.gpu_data(),
                                  difference_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             difference_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
