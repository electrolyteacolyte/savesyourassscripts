from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA1, SHA256
from Crypto.PublicKey import RSA
import os

def sign_data_with_cert(data):
    # Load the private key from the .cert file in the same directory as the script
    cert_file = os.path.join(os.path.dirname(__file__), 'your_private_key.cert')
    with open(cert_file, 'rb') as f:
        private_key_data = f.read()

    # Extract the private key and parse it as RSA
    private_key = RSA.import_key(private_key_data)

    # Hash the data with SHA-1
    hash_sha1 = SHA1.new(data)

    # Sign the hashed data with SHA-1
    signer_sha1 = pkcs1_15.new(private_key)
    signature_sha1 = signer_sha1.sign(hash_sha1)

    # Hash the data with SHA-256
    hash_sha256 = SHA256.new(data)

    # Sign the hashed data with SHA-256
    signer_sha256 = pkcs1_15.new(private_key)
    signature_sha256 = signer_sha256.sign(hash_sha256)

    return signature_sha1, signature_sha256

# Example usage
data_to_sign = b'ByteBloom'
signature_sha1, signature_sha256 = sign_data_with_cert(data_to_sign)

print("SHA-1 Signature:", signature_sha1.hex())
print("SHA-256 Signature:", signature_sha256.hex())
