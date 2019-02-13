#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-28 上午10:20
# @Author  : 林利芳
# @File    : crf.py
from scipy import optimize
import time
from model.module.feature import Feature
from utils.utils import *
from concurrent import futures
from scipy.misc import logsumexp

_gradient = None


def logsumexp_vec_mat(log_a, log_m):
	"""
	计算logsumexp log(e^x) = a-log(e^(-x+a)) np.dot(a, M) = exp(log a + log m)
	"""
	return logsumexp(log_a[:, np.newaxis] + log_m, axis=0)


def logsumexp_mat_vec(log_m, logb):
	"""np.dot(M, b) = exp(log M, log b)"""
	return logsumexp(log_m + logb, axis=1)


class CRF(object):
	def __init__(self, regtype=2, sigma=1.0, fd=5):
		"""
		CRF 初始化
		:param regtype: 正则化类型0,1,2 L1正则(loss + |w|/sigma), L2正则(loss + |w|^2/(2*sigma^2))
		:param sigma: 正则化系数
		:param fd: 特征频次
		"""
		self.sigma = sigma
		self.regtype = regtype
		self.feature = Feature(fd=fd)
	
	def fit(self, x_train, y_train, model_file=None, template_file=None, max_iter=20, n_jobs=None):
		"""
		训练模型
		:param x_train: x
		:param y_train: label
		:param template_file: 模板
		:param model_file: 模型文件
		:param max_iter: 迭代次数
		:param n_jobs: 进程数
		:return:
		"""
		self.feature(x_train, y_train, template_file)
		del x_train, y_train
		theta = random_param(self.feature.f_size)
		if not n_jobs:
			n_jobs = os.cpu_count() - 1
		likelihood = lambda x: -self.likelihood_parallel(x, n_jobs)
		likelihood_deriv = lambda x: -self.gradient_likelihood(x)
		start_time = time.time()
		print('L-BFGS 训练...')
		theta, _, _ = optimize.fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_deriv, maxiter=max_iter)
		if model_file:
			self.save_model(model_file, theta)
		print("训练耗时:\t{}s\n".format(int(time.time() - start_time)))
	
	def predict(self, x_test, model_file=None):
		"""
		预测结果
		:param x_test:
		:param model_file:
		:return:
		"""
		theta = self.load_model(model_file)
		seq_lens = [len(x) for x in x_test]
		y2label = dict([(self.feature.oby_dict[key], key) for key in self.feature.oby_dict.keys()])
		node_fs, edge_fs = self.feature.cal_observe_on(x_test)
		max_ys = self.tagging(seq_lens, node_fs, edge_fs, y2label, theta)
		return max_ys
	
	def tagging(self, seq_lens, node_fs, edge_fs, y2label, theta):
		"""
		动态规划计算序列状态
		:param seq_lens: [10,8,3,10] 句子长度
		:param node_fs: u特征
		:param edge_fs: b特征
		:param y2label: y2label id-label
		:param theta: 参数
		:return:
		"""
		bf_size = self.feature.bf_size
		theta_b = theta[0:bf_size]
		theta_u = theta[bf_size:]
		max_ys = []
		for seq_id, seq_len in enumerate(seq_lens):
			matrix_list = self.log_matrix(node_fs[seq_id], edge_fs[seq_id], theta_u, theta_b, self.feature.num_k)
			max_alpha = np.zeros((len(matrix_list), self.feature.num_k))
			
			max_index = []
			for loc_id, matrix in enumerate(matrix_list):
				if loc_id == 0:
					# 起始状态
					max_alpha[loc_id] = matrix[0, :]
				else:
					# 取状态概率最大的序列(num_k,num_k)
					at = max_alpha[loc_id - 1][:, np.newaxis] + matrix
					max_alpha[loc_id] = at.max(axis=0)
					max_index.append(at.argmax(axis=0))  # 索引代表前一时刻和当前时刻求和的最大值
			# 最终状态 取概率最大一个最大最终序列结果
			max_state = []
			ty = max_alpha[-1].argmax()
			max_state.append(y2label.get(ty, 'O'))
			# 反向追踪路径
			for a in (reversed(max_index)):
				max_state.append(y2label.get(a[ty], 'O'))
				ty = a[ty]
			max_ys.append(max_state[::-1])
		return max_ys
	
	@staticmethod
	def gradient_likelihood(theta):
		"""
		梯度-全局变量 dummy function
		:param theta: 参数
		:return:
		"""
		global _gradient
		return _gradient
	
	def likelihood_parallel(self, theta, n_jobs):
		"""
		并行计算参数 损失函数likelihood 梯度grad
		:param theta: 参数
		:param n_jobs: 进程数
		:return:
		"""
		global _gradient
		grad = np.array(self.feature.fss, copy=True)
		likelihood = np.dot(self.feature.fss, theta)
		seq_lens = self.feature.seq_lens
		seq_num = len(seq_lens)
		node_fs = self.feature.node_fs
		edge_fs = self.feature.edge_fs
		
		n_thread = 2 * n_jobs
		chunk = seq_num / n_thread
		chunk_id = [int(kk * chunk) for kk in range(n_thread + 1)]
		jobs = []
		with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
			for ii, start in enumerate(chunk_id[:-1]):
				end = chunk_id[ii + 1]
				job = executor.submit(self.likelihood, theta, node_fs[start:end], edge_fs[start:end])
				jobs.append(job)
		for job in jobs:
			_likelihood, _grad = job.result()
			likelihood += _likelihood
			grad += _grad
		# 正则化
		grad -= self.regularity_deriv(theta)
		_gradient = grad
		return likelihood - self.regularity(theta)
	
	def likelihood(self, theta, node_fs, edge_fs):
		"""
		计算序列特征概率
		对数似然函数 L(theta) = theta * fss -sum(log Z)
		梯度 grad = fss - sum (exp(theta * f) * f)
		:param node_fs: u特征 [[[10,25,30],[45,394],[]],[]]
		:param edge_fs: b特征
		:param theta: 参数 shape=(uf_size + bf_size,)
		:return:
		"""
		grad = np.zeros(self.feature.f_size)
		
		bf_size = self.feature.bf_size
		num_k = self.feature.num_k
		likelihood = 0
		grad_b = grad[0:bf_size]
		grad_u = grad[bf_size:]
		theta_b = theta[0:bf_size]
		theta_u = theta[bf_size:]
		for seq_id, (nodes, edges) in enumerate(zip(node_fs, edge_fs)):
			matrix_list = self.log_matrix(nodes, edges, theta_u, theta_b, num_k)
			log_alphas = self.forward_alphas(matrix_list)
			log_betas = self.backward_betas(matrix_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((num_k, num_k))
			for loc_id, matrix in enumerate(matrix_list):
				if loc_id == 0:
					expect = np.exp(matrix + log_betas[loc_id] - log_z)
				elif loc_id < len(matrix_list):
					expect = np.exp(matrix + log_alphas[loc_id - 1][:, np.newaxis] + log_betas[loc_id] - log_z)
				state_expect = np.sum(expect, axis=0)
				# 最小化参数分布
				for fs_id in nodes[loc_id]:
					grad_u[fs_id:fs_id + num_k] -= state_expect
				for fs_id in edges[loc_id]:
					grad_b[fs_id:fs_id + num_k * num_k] -= expect.reshape((num_k * num_k))
		return likelihood, grad
	
	@staticmethod
	def log_matrix(nodes, edges, theta_u, theta_b, num_k):
		"""
		特征抽取 条件随机场矩阵形式 M_i = sum( theta * f )
		:param nodes: 序列u特征 shape =(seq_len,) [[1245,4665],[2,33,455],...]
		:param edges: 序列u特征  shape =(seq_len,)
		:param theta_u: u特征参数
		:param theta_b: b特征参数
		:param num_k: 状态数
		:return: num_k 阶矩阵 shape = (seq_len,num_k,num_k)
		"""
		matrix_list = []
		for loc_id, (node, edge) in enumerate(zip(nodes, edges)):
			fv = np.zeros((num_k, num_k))
			for f_id in node:
				fv += theta_u[f_id:f_id + num_k]
			for f_id in edge:
				fv += theta_b[f_id:f_id + num_k * num_k].reshape((num_k, num_k))
			matrix_list.append(fv)
		# 初始状态
		matrix_list[0][1:, :] = -float('inf')
		return matrix_list
	
	@staticmethod
	def forward_alphas(m_list):
		"""
		前向算法  alpha  = dot(alpha, M) = exp(log alpha + log M)
		:param m_list: 条件随机场矩阵形式 M_i = sum( theta * fss )
		:return:
		"""
		log_alpha = m_list[0][0, :]
		log_alphas = [log_alpha]
		for logM in m_list[1:]:
			log_alpha = logsumexp_vec_mat(log_alpha, logM)
			log_alphas.append(log_alpha)
		return log_alphas
	
	@staticmethod
	def backward_betas(m_list):
		"""
		后向算法 beta = dot(M, beta) = exp(log M + log beta)
		:param m_list: 条件随机场矩阵形式 M_i = sum( theta * fss )
		:return:
		"""
		log_beta = np.zeros_like(m_list[-1][0, :])
		log_betas = [log_beta]
		for logM in m_list[-1:0:-1]:
			log_beta = logsumexp_mat_vec(logM, log_beta)
			log_betas.append(log_beta)
		return log_betas[::-1]
	
	def regularity(self, theta):
		"""
		正则化 regtype=0,1,2  L1, L2 正则
		:param theta: 参数 shape = (f_size,) = (uf_size + bf_size,)
		:return:
		"""
		if self.regtype == 0:
			regular = 0
		elif self.regtype == 1:
			regular = np.sum(np.abs(theta)) / self.sigma
		else:
			v = 2 * self.sigma ** 2
			regular = np.sum(np.dot(theta, theta)) / v
		return regular
	
	def regularity_deriv(self, theta):
		"""
		正则化微分 regtype=0,1,2  L1, L2 正则
		:param theta: 参数 shape = (f_size,) = (uf_size + bf_size,)
		:return:
		"""
		if self.regtype == 0:
			regular_der = 0
		elif self.regtype == 1:
			regular_der = np.sign(theta) / self.sigma
		else:
			v = self.sigma ** 2
			regular_der = theta / v
		return regular_der
	
	def load_model(self, model_file):
		"""
		加载模型
		:param model_file:
		:return:
		"""
		theta, self.feature = load_model(model_file)
		return theta
	
	def save_model(self, model_file, theta):
		"""
		保存模型
		:param model_file:
		:param theta:
		:return:
		"""
		model = [theta, self.feature]
		save_model(model_file, model)
