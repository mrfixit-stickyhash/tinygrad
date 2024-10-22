from tinygrad.sha3_256 import sha3_256

def test_sha3_256():
    input_data = b"hello world"
    expected_hash = "9fe490ca691cbd433ff56d80431d3ef2d5d7d34d71e21def0723750ead9c0f04"
    
    output_hash = sha3_256(input_data).hex()
    assert output_hash == expected_hash, f"Test failed: {output_hash} != {expected_hash}"
    
    print("Test passed!")

if __name__ == "__main__":
    test_sha3_256()
