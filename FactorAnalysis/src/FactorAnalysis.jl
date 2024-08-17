module FactorAnalysis

using LinearAlgebra
using Statistics

function factor_analysis(X::Matrix{Float64}, k::Int64;
  η::Float64=0.01, max_iter::Int64=1000,
  tol::Float64=1e-6)
  n, p = size(X)

  Λ = randn(p, k)
  F = randn(n, k)

  # モデル: X = FΛ' + E where:
  #   X: n × p 行列 (データ)
  #   F: n × k 行列 (因子スコア)
  #   Λ: p × k 行列 (因子負荷量)
  #   E: n × p 行列 (誤差項)
  #
  # 目的関数: L = ‖X - FΛ'‖²
  #
  # ΔL/ΔΛ = -2F'(X - FΛ')
  #       = -2F'E
  #
  # ΔL/ΔF = -2(X - FΛ')Λ
  #       = -2EΛ

  for iter in 1:max_iter
    E = X - F * Λ'

    grad_Λ = -2F' * E
    grad_F = -2E * Λ

    Λ_new = Λ - η * grad_Λ'
    F_new = F - η * grad_F

    if norm(Λ_new - Λ) < tol && norm(F_new - F) < tol
      Λ, F = Λ_new, F_new
      break
    end

    Λ, F = Λ_new, F_new

    if iter % 100 == 0
      error = norm(X - F * Λ')
      println("Iteration $iter, Error: $error")
    end
  end

  E = X - F * Λ'
  Ψ = Diagonal(var(E, dims=1))

  return Λ, F, Ψ
end

end # module FactorAnalysis
