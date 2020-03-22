/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

#define EPSILON 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	std::default_random_engine gen;

	//Set the number of particles.
	num_particles = 50;

	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles to first position
	// (based on estimates of x, y, theta and their uncertainties
	// from GPS) and all weights to 1.
	for (int cnt = 0; cnt < num_particles;++cnt)
	{
		Particle m_particle;
		// Add random Gaussian noise to each particle.
		m_particle.x = dist_x(gen);
		m_particle.y = dist_y(gen);
		m_particle.theta = dist_theta(gen);
		m_particle.weight = 1.0;
		m_particle.id = cnt;

		particles.push_back(m_particle);
		weights.push_back(1.0);
	}


	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
	//Add measurements to each particle and add random Gaussian noise.
	std::default_random_engine gen;

	for(auto& particle : particles)
	{
		if (fabs(yaw_rate) < EPSILON)
		{
			particle.x += velocity * cos(particle.theta) * delta_t;
			particle.y += velocity * sin(particle.theta) * delta_t;
		}
		else
		{
			particle.x += (velocity / yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
			particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
			particle.theta += yaw_rate * delta_t;
		}
		std::normal_distribution<double> dist_x(0, std_pos[0]);
		std::normal_distribution<double> dist_y(0, std_pos[1]);
		std::normal_distribution<double> dist_theta(0, std_pos[2]);

		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each
	// observed measurement and assign the observed measurement to this
	// particular landmark.

	for (auto& obs : observations)
	{
		double min_dist = std::numeric_limits<double>::max();
		for (auto pred : predicted)
		{
			if (dist(pred.x,pred.y,obs.x,obs.y) < min_dist)
			{
				obs.id = pred.id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

	double w_sum = 0.0;

	for (auto& particle : particles)
	{
		// The observations are given in the VEHICLE'S coordinate system.
		// We need to convert them into map coordinates.
		vector <LandmarkObs> transformed_obs;
		for (auto obs : observations)
		{
			obs.x = (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y) + particle.x;
			obs.y = (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y) + particle.y;
			transformed_obs.push_back(obs);
		}

		// Finding the landmarks which are in the particle sensor range
		vector<LandmarkObs> particle_landmarks_inrange;
		for (auto map_lm : map_landmarks.landmark_list)
		{
			if (dist(map_lm.x_f,map_lm.y_f,particle.x,particle.y) < sensor_range)
			{
				particle_landmarks_inrange.push_back({map_lm.id_i,map_lm.x_f,map_lm.y_f});
			}
		}

		// Associate landmarks to observations
		dataAssociation(particle_landmarks_inrange,transformed_obs);

		// Update the weights of each particle using a mult-variate Gaussian distribution.

		particle.weight = 1.0;
		for (auto obs : transformed_obs)
		{
			//Map::single_landmark_s sn_lm = *std::find(begin(map_landmarks.landmark_list),end(map_landmarks.landmark_list),[](Map::single_landmark_s sn_lm){return (sn_lm.id_i == obs.id);});
			Map::single_landmark_s sn_lm = map_landmarks.landmark_list[obs.id];
			particle.weight *= multiv_prob(std_landmark[0],std_landmark[1],
					obs.x,obs.y,
					sn_lm.x_f,sn_lm.y_f);
		}

		w_sum += particle.weight;
	}

	//normalizing the weights
	for(auto& particle : particles)
	{
		particle.weight = particle.weight / w_sum;
		weights[particle.id] = particle.weight;
	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional
	// to their weight.
	std::default_random_engine gen;
	std::uniform_int_distribution<int> int_uniform_dist(0,num_particles - 1);
	int index = int_uniform_dist(gen);
	double max_weight = *std::max_element(begin(weights),end(weights));
	double beta = 0.0;
	vector<Particle> resampled_particles;
	for(int cnt = 0; cnt < num_particles; ++cnt)
	{
		std::uniform_real_distribution<double> double_uniform_dist(0,2.0*max_weight);
	    beta += double_uniform_dist(gen);
	    while( beta > weights[index]) {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
