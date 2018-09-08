"""
Convolution Calculator
"""

W1 = 62
H1 = 62
D1 = 16

# ====================== CONVOLUTION LAYER =========================
conv_input = [W1, H1, D1]
K = 20
F = 4
S = 1
P = 0

W2 = (W1 - F + 2*P) / S + 1
H2 = (H1 - F + 2*P) / S + 1
D2 = K

conv_output = [int(W2), int(H2), int(D2)]
#print(conv_output)


# ========================== POOLING ================================
pool_input = [W1, H1, D1]

F = 2
S = 2

W2 = (W1 - F) / S + 1
H2 = (H1 - F) / S + 1
D2 = D1

pool_output = [int(W2), int(H2), int(D2)]
#print(pool_output)


# ================== TRANSVERSE CONVOLUTION LAYER ====================
transconv_input = [W1, H1, D1]
K = 25
F = 3
S = 1
P = 0
outP = 0

W2 = (W1 - 1) * S - 2*P + F + outP
H2 = (H1 - 1) * S - 2*P + F + outP
D2 = K

transconv_output = [int(W2), int(H2), int(D2)]
print(transconv_output)


# =========================== UNPOOLING =============================
unpool_input = [W1, H1, D1]
F = 2
S = 2
P = 0


W2 = (W1 - 1) * S - 2*P + F
H2 = (H1 - 1) * S - 2*P + F
D2 = D1

unpool_output = [int(W2), int(H2), int(D2)]
#print(unpool_output)