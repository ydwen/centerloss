#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const int num_output = this->layer_param_.center_loss_param().num_output();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize and fill the centers
    vector<int> center_shape(2); //center_shape(N_, K_);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> shape_1_X_N(1, N_);
  count_.Reshape(shape_1_X_N);
  difference_.ReshapeLike(*bottom[0]);
  vector<int> shape_1_X_M(1, M_);
  propagate_.Reshape(shape_1_X_M);
  top[1]->ReshapeLike(*top[0]);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // difference, count
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* count_data = count_.mutable_cpu_data();
  Dtype* difference_data = difference_.mutable_cpu_data();
  caffe_set(N_, (Dtype)0., count_data);
  for (int m = 0; m < M_; m++) {
    const int label_value = static_cast<int>(label[m]);
    count_data[label_value]++;
    caffe_sub(K_, bottom_data + m * K_, center + label_value * K_, difference_data + m * K_);
  }

  // propagate
  if (this->layer_param_.center_loss_param().type()==CenterLossParameter_DistanceType_L2) {
    const Dtype* difference = difference_.cpu_data();
    const Dtype margin_ = this->layer_param_.center_loss_param().margin();
    Dtype* propagate_data = propagate_.mutable_cpu_data();
    for (int m = 0; m < M_; m++) {
      Dtype d_norm = sqrt(caffe_cpu_dot(K_, difference + m * K_, difference + m * K_));
      propagate_data[m] = (d_norm - margin_) / d_norm;
      propagate_data[m] = propagate_data[m] > 0. ? propagate_data[m] : 0.;
    }
  } else if (this->layer_param_.center_loss_param().type()==CenterLossParameter_DistanceType_COSINE) {
    const Dtype eps_ = this->layer_param_.center_loss_param().eps();
    const Dtype margin_ = this->layer_param_.center_loss_param().margin();
    Dtype* propagate_data = propagate_.mutable_cpu_data();
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      // margin, xx, xc, cc
      Dtype g = margin_;
      Dtype r = caffe_cpu_dot(K_, bottom_data + m * K_, bottom_data + m * K_);
      Dtype s = caffe_cpu_dot(K_, bottom_data + m * K_, center + label_value * K_);
      Dtype t = caffe_cpu_dot(K_, center + label_value * K_, center + label_value * K_);
      if (!(s / sqrt(r) / sqrt(t) < g)) {
        propagate_data[m] = (Dtype)0.;
      } else {
        // a, b, c
        Dtype c = s * s - g * g * r * t;
        Dtype b = (Dtype)2. * (((Dtype)1. - g * g) * s * t - c);
        Dtype a = ((Dtype)1. - g * g) * t * (t - (Dtype)2. * s) + c;
        // x1, x2, propagate
        Dtype x1 = (-b + sqrt(b * b - (Dtype)4. * a * c + eps_)) / ((Dtype)2. * a);
        Dtype x2 = (-b - sqrt(b * b - (Dtype)4. * a * c + eps_)) / ((Dtype)2. * a);
        if (x1*x2 < (Dtype)0. || x1+x2 > (Dtype)1.) {
          propagate_data[m] = (x1 > (Dtype)0. && x1 < (Dtype)1.000001) ? x1 : x2;
        } else {
          propagate_data[m] = x1 > x2 ? x1 : x2;
        }
      }
    }
  }

  // propagated difference
  const Dtype* propagate = propagate_.cpu_data();
  for (int m = 0; m < M_; m++) {
    caffe_scal(K_, propagate[m], difference_data + m * K_);
  }
  // loss 
  Dtype mean_propagate = caffe_cpu_asum(M_, propagate_.cpu_data());
  top[0]->mutable_cpu_data()[0] = caffe_cpu_dot(M_ * K_, difference_.cpu_data(),
                                                         difference_.cpu_data()) / M_ / (Dtype)2.;
  top[1]->mutable_cpu_data()[0] = mean_propagate;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Gradient with respect to center
  if (this->param_propagate_down_[0]) {
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* difference = difference_.cpu_data();
    const Dtype* count = count_.cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      caffe_axpy(K_,  -(Dtype)1. / count[label_value], difference + m * K_, center_diff + label_value * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    caffe_cpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, difference_.cpu_data(), bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << "Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
