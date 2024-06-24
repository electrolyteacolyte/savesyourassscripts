from Crypto.PublicKey import RSA

# Generate a new RSA key pair
key = RSA.generate(2048)

# Get the private key in PEM format
private_key_pem = key.export_key()

# Save the private key to a file
with open('random_private_key.cer', 'wb') as f:
    f.write(private_key_pem)

print("Private key generated and saved to 'random_private_key.cer'")
