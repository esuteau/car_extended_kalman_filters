#include "kalman_filter.h"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Update state vector and covariance matrix
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	// Calculate the new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // With the extended kalman filter we use the real non linear function H(z) to
  // Calculate the new estimate.

  // Use H(x) to calculate the new prediction z_pred and then y
  // This function maps the location x from cartesian coordinates (px, py, vx, vy) to polar coordinates (rho, phi, rho_dot)
  // TODO: Check for division by zero
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  //cout << "(px, py, vx, vy) = " << px << ", " << py << ", " << vx << ", " << vy << endl;
  float z_pred_0 = sqrt(pow(px, 2) + pow(py, 2));
  float z_pred_1 = atan2(py, px);
  //cout << "z_pred_1 = " << z_pred_1 << endl;
  float z_pred_2 = (px * vx + py * vy) / z_pred_0;
  VectorXd z_pred(3);
  z_pred << z_pred_0, z_pred_1, z_pred_2;

  // Calculate y
  VectorXd y = z - z_pred;
  //cout << "y (angle) = " << y[1] << endl;

  // Make sure y is in the range [-pi; pi]
  while (y[1] < -M_PI){
    y[1] += 2 * M_PI;
  }
  while (y[1] > M_PI){
    y[1] -= 2 * M_PI;
  }
  
  // Use Jacobian matrix to calculate S and K
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

  // Calculate the new estimate x
	x_ = x_ + (K * y);

  // Use Jacobian matrix to calculate P
  long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}
