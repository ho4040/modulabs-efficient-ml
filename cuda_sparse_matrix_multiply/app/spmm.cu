#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <library_types.h>      
#include <cstdio>
#include <cstdlib>

int main(void) {
    printf("희소행렬 곱하기 테스트\n");

    // 이 예제는 다음 문서를 보고 작성되었습니다.
    // https://docs.nvidia.com/cuda/cusparselt/getting_started.html#cusparselt-workflow
    // https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/

    // cusparseLt 핸들을 초기화 합니다.
    cusparseLtHandle_t             handle;
    cusparseLtInit(&handle);

    // 행렬의 성질들을 정의합니다.

    // 크기
    constexpr int m            = 32;
    constexpr int n            = 32;
    constexpr int k            = 32;

    // 메모리상의 배치를 정의합니다.
    auto          order        = CUSPARSE_ORDER_ROW;
    auto          opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type         = CUDA_R_16F; // CUDA_R_16F, CUDA_R_16BF, CUDA_R_I8, CUDA_R_32F
    auto          compute_type = CUSPARSE_COMPUTE_32F;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);

    // 행렬 A : m x k
    // auto     num_A_rows     = m;
    // auto     num_A_cols     = k;
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;

    // 행렬 B : k x n
    // auto     num_B_rows     = k;
    // auto     num_B_cols     = n;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;

    // 행렬 C : m x n
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;

    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;

    // 메모리상의 크기를 계산합니다.
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(__half);
    
    __half hA[m * k];
    __half hB[k * n];
    __half hC[m * n];

    // 랜덤한 값으로 채웁니다.
    for (int i = 0; i < m * k; i++)
        hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    for (int i = 0; i < m * n; i++)
        hC[i] = static_cast<__half>(0.0f);


    // 디바이스 메모리를 준비합니다.
    __half *dA, *dB, *dC, *dD;
    cudaMalloc((void**) &dA, A_size); 
    cudaMalloc((void**) &dB, B_size);
    cudaMalloc((void**) &dC, C_size);
    dD = dC;

    // 호스트에서 디바이스로 데이터를 복사합니다.
    cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice);

    cusparseLtMatDescriptor_t      matA, matB, matC;
    
    // 압축 할 행렬을 만듭니다.
    cusparseLtStructuredDescriptorInit( // https://docs.nvidia.com/cuda/cusparselt/functions.html#cusparseltstructureddescriptorinit
        &handle,  
        &matA, 
        num_A_rows, 
        num_A_cols,
        lda, // Leading dimension of A (num_A_cols)  // https://docs.nvidia.com/cuda/cusparse/index.html#dense-matrix-format        
        alignment, // Memory alignment in bytes
        type, 
        order, // Memory layout (CUSPARSE_ORDER_COL || CUSPARSE_ORDER_ROW)
        CUSPARSELT_SPARSITY_50_PERCENT // Matrix sparsity ratio
    );

    // 밀집 행렬을 만듭니다.
    cusparseLtDenseDescriptorInit(
        &handle, &matB, num_B_rows,
        num_B_cols, ldb, alignment,
        type, order);
    
    // 계산 결과를 저장할 행렬을 만듭니다.
    cusparseLtDenseDescriptorInit(
        &handle, &matC, num_C_rows,
        num_C_cols, ldc, alignment,
        type, order);
    
    // 행렬 곱을 정의합니다.
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulDescriptorInit( // https://docs.nvidia.com/cuda/cusparselt/functions.html#cusparseltmatmuldescriptorinit
        &handle, 
        &matmul, 
        opA, 
        opB,
        &matA, //The structured matrix descriptor can used for matA or matB but not both.
        &matB, 
        &matC, 
        &matC,
        compute_type);
    
    // 알고리즘을 선택합니다.
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulAlgSelectionInit(
        &handle, 
        &alg_sel, 
        &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT
    );

    // 행렬 곱을 위한 플랜을 만듭니다.
    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulPlanInit(
        &handle, 
        &plan, 
        &matmul, 
        &alg_sel
    );

    cudaStream_t stream = nullptr;
    //--------------------------------------------------------------------------
    // 행렬 A를 2:4 구조적 희소성을 가지도록 프루닝을 수행합니다.
    cusparseLtSpMMAPrune(
        &handle, 
        &matmul, 
        dA, 
        dA, 
        CUSPARSELT_PRUNE_SPMMA_TILE,
        stream
    );
    
    // 2:4 sparsity 를 가지도록 행렬이 프루닝 되었으므로
    // 프루닝 된 부분을 제외한 행렬의 크기를 구합니다.

    size_t compressed_size, compresse_buffer_size;    
    void *dA_compressedBuffer;
    __half *dA_compressed;
    cusparseLtSpMMACompressedSize(&handle, 
        &plan, 
        &compressed_size, 
        &compresse_buffer_size // 행렬 압축에 필요한 임시버퍼 사이즈를 알려줌. 
    );
    cudaMalloc((void**) &dA_compressed, compressed_size); // 압축된 행렬을 저장할 디바이스 메모리를 할당합니다.
    cudaMalloc((void**) &dA_compressedBuffer, compresse_buffer_size); // 압축에 필요한 임시버퍼를 할당합니다.
    printf("행렬 A 크기: A_height[%d] * lda[%d] * sizeof(__half)[%zu] = %zu Bytes\n", A_height, lda, sizeof(__half), A_size);
    printf("압축된 행렬의 크기: %zu Bytes\n", compressed_size);

    // 압축된 행렬을 만듭니다.
    cusparseLtSpMMACompress(&handle, 
        &plan, 
        dA, 
        dA_compressed, 
        dA_compressedBuffer, 
        stream);

    // 알고리즘을 지정합니다.
    int alg = 0;
    cusparseLtMatmulAlgSetAttribute(
        &handle, &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID, 
        &alg,  // Possible Values : 0, MAX
        sizeof(alg));
    
    // 행렬 곱을 위한 플랜을 초기화 합니다.
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
    
    // 플렌에 필요한 워크스페이스 크기를 계산하고 디바이스 메모리를 할당합니다.
    size_t workspace_size;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    void* d_workspace;
    cudaMalloc((void**) &d_workspace, workspace_size);

    int  num_streams = 0;
    cudaStream_t* streams = nullptr; //병렬처리X
    
    // 다음 수식에 맞춰 계산을 수행.    
    // D = Activation(alpha*(op(a)*op(b)) + beta*c) * scale
    // https://docs.nvidia.com/cuda/cusparselt/functions.html#cusparseltmatmul
    float alpha = 1.0f; 
    float beta  = 0.0f; 
    cusparseLtMatmul( 
        &handle, 
        &plan, 
        &alpha,  // scalar/vector of scalars used for multiplication
        dA_compressed, 
        dB,
        &beta, 
        dC, 
        dD, 
        d_workspace, 
        streams,  // Pointer to CUDA stream array for the computation
        num_streams // Number of CUDA streams in
    );

    // 자원 릴리즈
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);

    
    // 디바이스에서 호스트로 데이터 가져옴
    cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost);

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);

    // 호스트에 있는 A*B 값을 계산해서 hC_result에 저장
    float hC_result[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < k; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = sum;  // [i][j]
        }
    }

    // 호스트와 디바이스의 결과를 비교
    int correct = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            auto host_value   = hC_result[pos];
            if (device_value != host_value) {
                // direct floating point comparison is not reliable
                printf("원소(%d, %d):\t 호스트[%f] vs 디바이스[%f]\n", i, j, host_value, device_value);
                correct = 0;
                break;
            }
        }
    }

    if (correct)
        printf("결과: 정상\n");
    else
        printf("결과: 비정상\n");
        
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dA_compressed);
    cudaFree(d_workspace);
    cudaFree(dA_compressedBuffer);

    printf("종료\n");

    return EXIT_SUCCESS;


}   