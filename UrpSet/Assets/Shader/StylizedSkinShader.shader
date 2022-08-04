Shader "ADA/Character/StylizedSkinShader"
{
    Properties
    {
        [Toggle] _Visibillity ("Visibillity", float) = 1
        [Toggle]_Optimize ("Optimize", float) = 0

        [Space(20)]
        [HDR]_Color ("Color", Color) = (1,1,1,1)
        _BaseMap ("Main Texture (RGB : Albedo Texture, A : Alpha) ", 2D) = "white" {}
        
        _ReceiveShadowAmount ("Receive Shadow Amount", Range(0,1)) = 1

        [Header(Normal)]
        [Space(10)]
        [NoScaleOffset]_NormalTex ("Normal Texture", 2D) = "bump" {}
        _NormalStrength ("Normal Strength", float) = 1
        [Toggle]_NormalGFlip("Normal G Channel Flip", float) = 0

        [Space(20)]
        [Header(Stylized Diffuse)]
        [Space(10)]
        _MidColor("Mid Tone Color", Color) = (1,1,1,1)
        _MidThreshold ("Mid Tone Threshold", Range(0,1)) = 1
        _MidSmooth ("Mid Tone Smooth", Range(0,0.5)) = 0.25
        
        [Space(10)]
        _ShadowColor ("Shadow Color", Color) = (0,0,0,1)
        _ShadowThreshold ("Shadow Threshold", Range(0,1)) = 0.5
        _ShadowSmooth ("Shadow Smooth", Range(0,0.5)) = 0.25
        
        [Space(10)]
        _ReflectColor ("Reflect Color", Color) = (0,0,0,1)
        _ReflectThreshold ("Reflect Threshold", Range(0,1)) = 0
        _ReflectSmooth ("Reflect Smooth", Range(0,0.5)) = 0.25
        [Space(10)]
        
        _GIIntensity("GI Intensity", Range(0,2)) = 1

        [Space(20)]

        [Header(Dyeing)]
        _DyeingTex("Dyeing Texture", 2D) = "black" {}
        _DyeingColor1("Dyeing Color 1 (R)", Color) = (1,1,1,1)
        _DyeingRoughness1("Dyeing Roughness 1", Range(0,1)) = 1
        _DyeingArea1("Dyeing Area 1", Range(0.1,0.9)) = 0.5
        _DyeingAmount1("Dyeing Amount 1", Range(0,1)) = 1
        _DyeingColor2("Dyeing Color 2 (G)", Color) = (1,1,1,1)
        _DyeingRoughness2("Dyeing Roughness 2", Range(0,1)) = 1
        _DyeingArea2("Dyeing Area 2", Range(0.1,0.9)) = 0.5
        _DyeingAmount2("Dyeing Amount 2", Range(0,1)) = 1
        _DyeingColor3("Dyeing Color 3 (B)", Color) = (1,1,1,1)
        _DyeingRoughness3("Dyeing Roughness 3", Range(0,1)) = 1
        _DyeingArea3("Dyeing Area 3", Range(0.1,0.9)) = 0.5
        _DyeingAmount3("Dyeing Amount 3", Range(0,1)) = 1
        _DyeingColor4("Dyeing Color 4 (A)", Color) = (1,1,1,1)
        _DyeingRoughness4("Dyeing Roughness 4", Range(0,1)) = 1
        _DyeingArea4("Dyeing Area 4", Range(0.1,0.9)) = 0.5
        _DyeingAmount4("Dyeing Amount 4", Range(0,1)) = 0
        
        [Space(10)]
        _DyeingTex2("Dyeing Texture 2", 2D) = "black" {}
        _DyeingColor5("Dyeing Color 5 (R)", Color) = (1,1,1,1)
        _DyeingRoughness5("Dyeing Roughness 5", Range(0,1)) = 1
        _DyeingArea5("Dyeing Area 5", Range(0.1,0.9)) = 0.5
        _DyeingAmount5("Dyeing Amount 5", Range(0,1)) = 1
        _DyeingColor6("Dyeing Color 6 (G)", Color) = (1,1,1,1)
        _DyeingRoughness6("Dyeing Roughness 6", Range(0,1)) = 1
        _DyeingArea6("Dyeing Area 6", Range(0.1,0.9)) = 0.5
        _DyeingAmount6("Dyeing Amount 6", Range(0,1)) = 1
        
    
        [Space(20)]
        [Header(Stylized Reflection)]
        [Space(10)]

        [NoScaleOffset] _MaskTex ("Mask Texture ( R:Smoothness, G:Specular, B:AO, A:Emission)", 2D) = "white" {}
        _Smoothness ("Smoothness", Range(0,1)) = 1
        _Metallic ("Metallic ", Range(0,1)) = 1
        
        _AoStrength("AO Strength", Range(0, 1)) = 1.0

        [Space(10)]
        _EmissionColor ("Emission Color", Color) = (0,0,0,0)
        _EmissionEdgeSmooth ("Emission Edge Smooth ", float) = 1
        _EmissionIntensity ("Emission Intensity", float) = 0

        
        [Space(10)]
        [Toggle] _UseGGX ("Use GGX", float) = 0
        _SpecColor("Specular Color", Color) = (0.5, 0.5, 0.5)
        _SpecularThreshold("Specular Threshold", Range(0.1,2)) = 0.5
        _SpecularSmooth ("Specular Smooth", Range(0,0.5)) = 0.5
        _SpecularIntensity("Specular Intensity", Range(0,10)) = 1
        [Space(10)]
        [Toggle] _DirectionalFresnel ("Directional Fresnel", float) = 0
        
        _FresnelColor("Fresnel Color", Color) = (1,1,1,1)
        _FresnelThreshold("Fresnel Threshold", Range(0,1)) = 0.8
        _FresnelSmooth("Fresnel Smooth", Range(0,0.5)) = 0.25
        _FresnelIntensity ("Non Metal Fresnel Intensity", float) = 1
        _MetalFresnelIntensity("Metal Fresnel Intensity", float) = 1

        [Space(10)]
        _ReflProbeColor ("Reflection Probe Color", Color) = (1,1,1,1)
        _ReflProbeIntensity("Non Metal Reflection Probe Intensity", float) = 1
        _MetalReflProbeIntensity ("Metal Reflection Probe Intensity", float) = 1
        _ReflProbeBlurAmount("Reflection Probe Blur Amount", Range(0,2)) = 1

        [Header(SSS)]
        _SSSTex("SSS Texture", 2D) = "black" {}
        _SSSColor ("SSS Color", color) = (0,0,0,0)
        _Attenuation ("Attenuation", Range(0,2)) = 1
        _SSSAmbient ("Ambient", Range(0,1)) = 0
        
        _Distortion ("Distortion", Range(-1,1)) = 1
        _SSSPower ("SSS Power", Range(0.001,5)) = 1
        _SSSScale ("SSS Scale", Range(0,20)) = 1


   
        [Space(20)]
        [Header(Alpha Culling)]
        [Space(10)]
        [Toggle] _AlphaTest ("AlphaTest", float) = 0
        _AlphaCutout ("Alpha Cutout", Range(0,1)) = 0
        [Enum(UnityEngine.Rendering.CullMode)]_Cull ("Culling", float) = 2

        
        [HideInInspector][NoScaleOffset]unity_Lightmaps("unity_Lightmaps", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_LightmapsInd("unity_LightmapsInd", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_ShadowMasks("unity_ShadowMasks", 2DArray) = "" {}

        //------------------------------------------------------
        
    }
    SubShader
    {
        Tags
        {
            "RenderPipeline"="UniversalPipeline"
            "RenderType"="Opaque"
            "Queue"="Geometry+0"
        }
        LOD 100

        Pass
        {
            Name "Universal Forward"
            Tags 
            { 
                "LightMode" = "UniversalForward"
            }
            Cull [_Cull]
            ZTest LEqual


            HLSLPROGRAM
            #pragma prefer_hlslcc gles
            #pragma exclude_renderers d3d11_9x
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            
             // GPU Instancing
            #pragma multi_compile_instancing
            #pragma multi_compile_fog


            // Recieve Shadow
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _SHADOWS_SOFT

            #pragma multi_compile _ _ALPHATEST_ON
            #pragma multi_compile _ _OPTIMIZE_ON
                        
            
            

            TEXTURE2D(_BaseMap);   SAMPLER(sampler_BaseMap);   
            TEXTURE2D(_NormalTex);   SAMPLER(sampler_NormalTex);  
            TEXTURE2D(_MaskTex);   SAMPLER(sampler_MaskTex); 
            TEXTURE2D(_SSSTex);   SAMPLER(sampler_SSSTex); 
            TEXTURE2D(_DyeingTex);  SAMPLER(sampler_DyeingTex);
            TEXTURE2D(_DyeingTex2);  SAMPLER(sampler_DyeingTex2);

            CBUFFER_START(UnityPerMaterial)    

            float4 _BaseMap_ST;
            half4 _MaskTex_ST;
            
            float4 _Color,_FresnelColor;
            
            float _ReceiveShadowAmount;
            float _NormalStrength,_NormalGFlip;
            
            float _Metallic;
            float _Smoothness;
            float _AoStrength;


            float _AlphaCutout;

            float4 _EmissionColor;
            float _EmissionEdgeSmooth;
            float _EmissionIntensity;
            float _EmissionMode;


            float4 _MidColor, _ShadowColor, _ReflectColor, _ReflProbeColor;
            float _MidThreshold, _MidSmooth, _ShadowThreshold, _ShadowSmooth, _ReflectThreshold, _ReflectSmooth;
            float _SpecularThreshold, _SpecularSmooth, _SpecularIntensity ,_FresnelIntensity, _FresnelThreshold, _FresnelSmooth, _DirectionalFresnel, _MetalFresnelIntensity;
            float _ReflProbeIntensity, _MetalReflProbeIntensity, _ReflProbeBlurAmount;
            float _GIIntensity;

            float _UseGGX;

            float4 _SSSColor;
            float _Attenuation, _Distortion, _SSSPower, _SSSScale, _SSSAmbient;

            float _Visibillity;

            float4 _DyeingColor1, _DyeingColor2, _DyeingColor3, _DyeingColor4, _DyeingColor5, _DyeingColor6;
            float _DyeingAmount1, _DyeingAmount2, _DyeingAmount3, _DyeingAmount4, _DyeingAmount5, _DyeingAmount6;
            float _DyeingRoughness1, _DyeingRoughness2, _DyeingRoughness3, _DyeingRoughness4, _DyeingRoughness5, _DyeingRoughness6;
            float _DyeingArea1, _DyeingArea2, _DyeingArea3, _DyeingArea4, _DyeingArea5, _DyeingArea6;



            CBUFFER_END

            

            struct appdata
            {
                float4 color : COLOR0;
                float4 vertex : POSITION;
                float2 texcoord : TEXCOORD0;
                float2 lightmapUV : TEXCOORD1;
                float3 normalOS : NORMAL;
                float4 tangentOS : TANGENT;
                UNITY_VERTEX_INPUT_INSTANCE_ID                                
            };


            struct v2f
            {
                float4 color : COLOR0;
                float4 positionCS : SV_POSITION;
                
                float2 uv                       : TEXCOORD0;
                DECLARE_LIGHTMAP_OR_SH(lightmapUV, vertexSH, 1);
                float3 positionWS               : TEXCOORD2;
                float3 normalWS                 : TEXCOORD3;
                float4 tangentWS                : TEXCOORD4;    // xyz: tangent, w: sign
                float3 viewDirWS                : TEXCOORD5;
                half4 fogFactorAndVertexLight   : TEXCOORD6; // x: fogFactor, yzw: vertex light
                float4 shadowCoord              : TEXCOORD7;
                //float3 viewDirTS                : TEXCOORD8;

                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
  
            };

            float4 RotateX(float4 localRotation, float angle)
            {
                float angleX = radians(angle);
                float c = cos(angleX);
                float s = sin(angleX);
                float4x4 rotateXMatrix  = float4x4( 1,  0,  0,  0,
                                                    0,  c,  -s, 0,
                                                    0,  s,  c,  0,
                                                    0,  0,  0,  1);
                return mul(localRotation, rotateXMatrix);
            }

            float4 RotateY(float4 localRotation, float angle)
            {
                float angleY = radians(angle);
                float c = cos(angleY);
                float s = sin(angleY);
                float4x4 rotateYMatrix  = float4x4( c,  0,  s,  0,
                                                    0,  1,  0,  0,
                                                    -s, 0,  c,  0,
                                                    0,  0,  0,  1);
                return mul(localRotation, rotateYMatrix);
            }

            float4 RotateZ(float4 localRotation, float angle)
            {
                float angleZ = radians(angle);
                float c = cos(angleZ);
                float s = sin(angleZ);
                float4x4 rotateZMatrix  = float4x4( c,  -s, 0,  0,
                                                    s,  c,  0,  0,
                                                    0,  0,  1,  0,
                                                    0,  0,  0,  1);
                return mul(localRotation, rotateZMatrix);
            }



            half LinearStep(half minValue, half maxValue, half In)
            {
                return saturate((In-minValue) / (maxValue - minValue));
            }

            float StrandSpecular(float4 T, float3 V, float3 L, float3 N, float shift, float exponent, float strength)
            {
                float3 Tan = T.xyz;//RotateZ(RotateY(RotateX( float4(T.xyz,1),_RotateTangent.x), _RotateTangent.y), _RotateTangent.z).xyz;  
                float3 Nor = N.xyz;
                float3 H = normalize(L+V);
                float3 B = -normalize(cross( Tan ,Nor) * T.w);
                B = normalize(B + Nor * shift );

                float dotBH =  dot(B, H);
                float sinBH = sqrt(1.0 - dotBH*dotBH );
                float dirAtten = smoothstep(-1, 0 , dotBH);
                return dirAtten * pow(sinBH, 1 / (exponent)+00001 ) * strength;
            }

            
            void InitializeInputData(v2f input, half3 normalTS, out InputData inputData)
            {
                inputData = (InputData)0;

                inputData.positionWS = input.positionWS;
        
                half3 viewDirWS = SafeNormalize(input.viewDirWS);

                float sgn = input.tangentWS.w;      // should be either +1 or -1
                float3 bitangent = sgn * cross(input.normalWS.xyz, input.tangentWS.xyz);
                inputData.normalWS = TransformTangentToWorld(normalTS, half3x3(input.tangentWS.xyz, bitangent.xyz, input.normalWS.xyz));
            
                inputData.normalWS = normalize(inputData.normalWS);
                inputData.viewDirectionWS = viewDirWS;

            #if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
                inputData.shadowCoord = input.shadowCoord;
            #elif defined(MAIN_LIGHT_CALCULATE_SHADOWS)
                inputData.shadowCoord = TransformWorldToShadowCoord(inputData.positionWS);
            #else
                inputData.shadowCoord = float4(0, 0, 0, 0);
            #endif

                inputData.fogCoord = input.fogFactorAndVertexLight.x;
                inputData.vertexLighting = input.fogFactorAndVertexLight.yzw;
                inputData.bakedGI = SAMPLE_GI(input.lightmapUV, input.vertexSH, inputData.normalWS);
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(input.positionCS);
                inputData.shadowMask = SAMPLE_SHADOWMASK(input.lightmapUV);
            }

            half3 DirectBDRFCustom(BRDFData brdfData, half3 normalWS, half3 lightDirectionWS, half3 viewDirectionWS, half specular)
            {

                float3 halfDir = SafeNormalize(float3(lightDirectionWS) + float3(viewDirectionWS));

                float NoH = saturate(dot(normalWS, halfDir));
                half LoH = saturate(dot(lightDirectionWS, halfDir));

                float d = NoH * NoH * brdfData.roughness2MinusOne + 1.00001f;

                half LoH2 = LoH * LoH;
                half specularTerm = brdfData.roughness2 / ((d * d) * max(0.1h, LoH2) * brdfData.normalizationTerm);

                
            #if defined (SHADER_API_MOBILE) || defined (SHADER_API_SWITCH)
                specularTerm = specularTerm - HALF_MIN;
                specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
            #endif
                half3 color = lerp(LinearStep( _SpecularThreshold - _SpecularSmooth, _SpecularThreshold + _SpecularSmooth, specularTerm  ), specularTerm, _UseGGX) * brdfData.specular * specular * max(0,_SpecularIntensity) + brdfData.diffuse; //LinearStep( _SpecularThreshold - _SpecularSmooth, _SpecularThreshold + _SpecularSmooth, specularTerm  ) * brdfData.specular * max(0,_SpecularIntensity) + brdfData.diffuse;
                return color;

                

            }

            half3 CalculateRadiance(Light light, half3 normalWS, half3 brushStrengthRGB)
            {
                half NdotL = dot(normalWS, light.direction);

                half halfLambert = NdotL * 0.5 + 0.5;


                half smoothMidTone = LinearStep( _MidThreshold - _MidSmooth, _MidThreshold + _MidSmooth, halfLambert);
                half3 MidToneColor = lerp( _MidColor.rgb , 1 , smoothMidTone);
                
                half smoothShadow = LinearStep ( _ShadowThreshold - _ShadowSmooth, _ShadowThreshold + _ShadowSmooth, halfLambert ) * (lerp(1,light.distanceAttenuation * light.shadowAttenuation,_ReceiveShadowAmount) ) ;
                half3 ShadowColor = lerp( _ShadowColor.rgb , MidToneColor, smoothShadow );   
                half smoothReflect = LinearStep( _ReflectThreshold - _ReflectSmooth, _ReflectThreshold + _ReflectSmooth, halfLambert);
                half3 ReflectColor = lerp(_ReflectColor.rgb , ShadowColor , smoothReflect);
                half3 radiance = light.color * ReflectColor;
                return radiance;
            }



            half3 LightingPhysicallyBasedCustom(BRDFData brdfData, half3 radiance, half3 lightDirectionWS, half3 normalWS, half3 viewDirectionWS, half3 positionWS, half specular)
            {             
            
                return DirectBDRFCustom(brdfData, normalWS, lightDirectionWS, viewDirectionWS, specular) * radiance;
            }

            half3 LightingPhysicallyBasedCustom(BRDFData brdfData, half3 radiance, Light light, half3 normalWS, half3 viewDirectionWS, half3 positionWS, half specular)
            {
                return LightingPhysicallyBasedCustom(brdfData, radiance, light.direction, normalWS, viewDirectionWS, positionWS , specular);
            }

            half3 EnvironmentBRDFCustom(BRDFData brdfData, half3 radiance, half3 indirectDiffuse, half3 indirectSpecular, half fresnelTerm)  
            {
                half3 c = indirectDiffuse * brdfData.diffuse * _GIIntensity;
                float surfaceReduction = 1.0 / (brdfData.roughness2 + 1.0);
                c += surfaceReduction * indirectSpecular * lerp(brdfData.specular * radiance, brdfData.grazingTerm * _FresnelColor.rgb, fresnelTerm);   
                return c;
            }



            half3 GlobalIlluminationCustom(BRDFData brdfData, half3 radiance, half3 bakedGI, half occlusion, half3 normalWS, half3 viewDirectionWS, half metallic, half ndotl)
            {
                half3 reflectVector = reflect(-viewDirectionWS, normalWS);
                half fresnelTerm = smoothstep(_FresnelThreshold - _FresnelSmooth, _FresnelThreshold + _FresnelSmooth,  1.0 - saturate(dot(normalWS, viewDirectionWS))) * max(0,lerp(_FresnelIntensity, _MetalFresnelIntensity, step(0.5,metallic) ) ) * ndotl;

                half3 indirectDiffuse = bakedGI * occlusion;
                half3 indirectSpecular = GlossyEnvironmentReflection(reflectVector, brdfData.perceptualRoughness * _ReflProbeBlurAmount, occlusion) * lerp(max(0,_ReflProbeIntensity), max(0,_MetalReflProbeIntensity), step(0.5,metallic) ) * _ReflProbeColor.rgb;

                return EnvironmentBRDFCustom(brdfData, radiance, indirectDiffuse, indirectSpecular, fresnelTerm);
            }



            half4 UniversalFragmentPBRCustom(InputData inputData, half3 albedo, half metallic, half3 specular, half smoothness, half occlusion, half3 emission, half alpha, half specularMask)
            {
                BRDFData brdfData;
                InitializeBRDFData(albedo, metallic, specular, smoothness, alpha, brdfData);
                
                inputData.shadowCoord = TransformWorldToShadowCoord(inputData.positionWS);
                Light mainLight = GetMainLight(inputData.shadowCoord);


                

                
                float3 radiance = CalculateRadiance(mainLight, inputData.normalWS, float3(0, 0, 0));


                MixRealtimeAndBakedGI(mainLight, inputData.normalWS, inputData.bakedGI, half4(0, 0, 0, 0));

                float ndotl = lerp(1, LinearStep( _ShadowThreshold - _ShadowSmooth, _ShadowThreshold + _ShadowSmooth,  dot(mainLight.direction, inputData.normalWS) * 0.5 + 0.5), _DirectionalFresnel);

                half3 color = GlobalIlluminationCustom(brdfData, radiance, inputData.bakedGI, occlusion, inputData.normalWS, inputData.viewDirectionWS, metallic, ndotl);
                color += LightingPhysicallyBasedCustom(brdfData, radiance, mainLight, inputData.normalWS, inputData.viewDirectionWS, inputData.positionWS, specularMask);

            #ifdef _ADDITIONAL_LIGHTS
                uint pixelLightCount = GetAdditionalLightsCount();
                for (uint lightIndex = 0u; lightIndex < pixelLightCount; ++lightIndex)
                {
                    Light light = GetAdditionalLight(lightIndex, inputData.positionWS);
                    color += LightingPhysicallyBased(brdfData, light, inputData.normalWS, inputData.viewDirectionWS);
                }
            #endif

            #ifdef _ADDITIONAL_LIGHTS_VERTEX
                color += inputData.vertexLighting * brdfData.diffuse;
            #endif

                color += emission;
                return half4(color, alpha);
            }


            float3 BlendOverlay(float3 base, float3 blend)
            {
                float3 check = step(0.5, base);
                float3 result = check * (half3(1, 1, 1) - ((half3(1, 1, 1) - 2 * (base - 0.5f)) * (1 - blend)));
                result += (1 - check) * (2 * base) * blend;
                return result;
            }



            v2f vert (appdata input)
            {
                v2f output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.vertex.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);

                half3 viewDirWS = GetWorldSpaceViewDir(vertexInput.positionWS);
                half3 vertexLight = VertexLighting(vertexInput.positionWS, normalInput.normalWS);
                //half fogFactor = ComputeFogFactor(vertexInput.positionCS.z);

                output.uv = TRANSFORM_TEX(input.texcoord, _BaseMap); 
                output.color = input.color;

                // already normalized from normal transform to WS.
                output.normalWS = normalInput.normalWS;
                output.viewDirWS = viewDirWS;
            
                real sign = input.tangentOS.w * GetOddNegativeScale();
                half4 tangentWS = half4(normalInput.tangentWS.xyz, sign);
           
            
                output.tangentWS = tangentWS;
            
                OUTPUT_LIGHTMAP_UV(input.lightmapUV, unity_LightmapST, output.lightmapUV);
                OUTPUT_SH(output.normalWS.xyz, output.vertexSH);

                //output.fogFactorAndVertexLight = half4(fogFactor, vertexLight);

                output.positionWS = vertexInput.positionWS;
                        
                output.shadowCoord = GetShadowCoord(vertexInput);
                output.positionCS = vertexInput.positionCS;
                output.fogFactorAndVertexLight = half4(ComputeFogFactor(vertexInput.positionCS.z), vertexLight);
                
                


                return output;
            }

            



            half4 frag (v2f i) : SV_Target
            {

                UNITY_SETUP_INSTANCE_ID(i);
                
               
                float3 NormalTS = UnpackNormal( SAMPLE_TEXTURE2D(_NormalTex, sampler_NormalTex, i.uv) ) * float3(_NormalStrength, _NormalStrength * lerp(1,-1,_NormalGFlip), 1);
                
                float4 albedo = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv) * _Color;

                float4 dyeingTex = SAMPLE_TEXTURE2D(_DyeingTex, sampler_DyeingTex, i.uv);
                float4 dyeingTex2 = SAMPLE_TEXTURE2D(_DyeingTex2, sampler_DyeingTex2, i.uv);

                dyeingTex.r = smoothstep(_DyeingArea1 - 0.1, _DyeingArea1 + 0.1, dyeingTex.r);
                dyeingTex.g = smoothstep(_DyeingArea2 - 0.1, _DyeingArea2 + 0.1, dyeingTex.g);
                dyeingTex.b = smoothstep(_DyeingArea3 - 0.1, _DyeingArea3 + 0.1, dyeingTex.b);
                dyeingTex.a = smoothstep(_DyeingArea4 - 0.1, _DyeingArea4 + 0.1, dyeingTex.a);
                dyeingTex2.r = smoothstep(_DyeingArea5 - 0.1, _DyeingArea5 + 0.1, dyeingTex2.r);
                dyeingTex2.g = smoothstep(_DyeingArea6 - 0.1, _DyeingArea6 + 0.1, dyeingTex2.g);

                albedo.rgb = lerp(lerp(lerp(lerp(albedo.rgb, BlendOverlay(albedo.rgb, _DyeingColor1.rgb), dyeingTex.r * _DyeingAmount1),BlendOverlay(albedo.rgb, _DyeingColor2.rgb), dyeingTex.g *_DyeingAmount2),BlendOverlay(albedo.rgb, _DyeingColor3.rgb), dyeingTex.b * _DyeingAmount3),BlendOverlay(albedo.rgb, _DyeingColor4.rgb), dyeingTex.a * _DyeingAmount4) ;
                albedo.rgb = lerp(lerp(albedo.rgb, BlendOverlay(albedo.rgb, _DyeingColor5.rgb), dyeingTex2.r * _DyeingAmount5),BlendOverlay(albedo.rgb, _DyeingColor6.rgb), dyeingTex2.g *_DyeingAmount6);


                //float3 albedoG = lerp(albedo.rgb, BlendOverlay(albedo.rgb, _DyeingColor2), _DyeingAmount2) ;
                //float3 albedoB = lerp(albedo.rgb, BlendOverlay(albedo.rgb, _DyeingColor3), _DyeingAmount3) ;
                
                //albedo.rgb = albedo.rgb * (1 - saturate(dyeingTex.r + dyeingTex.g + dyeingTex.b)) + albedoR + albedoG + albedoB;

                Light mainLight = GetMainLight();

                float4 col;

                InputData inputData;
                InitializeInputData(i, NormalTS, inputData);
                i.shadowCoord = TransformWorldToShadowCoord(i.positionWS);

                float4 mask = SAMPLE_TEXTURE2D(_MaskTex, sampler_MaskTex, i.uv);

                float metallic = _Metallic;
                float smoothness = mask.r * _Smoothness;

                float dyeingSmoothness = lerp(lerp(lerp(lerp(lerp(lerp( smoothness , 1-_DyeingRoughness1, dyeingTex.r * _DyeingAmount1),  1- _DyeingRoughness2, dyeingTex.g * _DyeingAmount2),  1- _DyeingRoughness3, dyeingTex.b * _DyeingAmount3),  1- _DyeingRoughness4, dyeingTex.a * _DyeingAmount4),  1- _DyeingRoughness5, dyeingTex2.r * _DyeingAmount5),  1- _DyeingRoughness6, dyeingTex2.g * _DyeingAmount6);

                
                float occlusion =  lerp( 1 , mask.b , _AoStrength);
                float specularMask = mask.g;
                
                

                float3 emissive =  pow( abs(mask.a), _EmissionEdgeSmooth) * _EmissionIntensity * _EmissionColor.rgb;

                float smoothShadow;
                #if _OPTIMIZE_ON
                    float4 mainTex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv * _BaseMap_ST.xy + _BaseMap_ST.zw ) * _Color;
                    mainLight = GetMainLight(inputData.shadowCoord);
                    col.rgb = mainTex * CalculateRadiance(mainLight, inputData.normalWS, float3(0, 0, 0));
                    col.rgb += col.rgb * SampleSH(inputData.normalWS);
                #else
                    col = UniversalFragmentPBRCustom(inputData, albedo.rgb, metallic, 0.5 , dyeingSmoothness , occlusion, emissive.rgb, albedo.a, specularMask);
                #endif
                col.rgb = MixFog(col.rgb, inputData.fogCoord);

                //SSS
                float4 thickness = SAMPLE_TEXTURE2D(_SSSTex, sampler_SSSTex, i.uv);

                float3 H = normalize( mainLight.direction + inputData.normalWS * _Distortion);
                float VdotH = pow(saturate(dot( inputData.viewDirectionWS ,-H)),_SSSPower) * _SSSScale; 
                float3 SSS = _Attenuation * (VdotH + _SSSAmbient) * thickness.x * _SSSColor.rgb;

                col.rgb += SSS;


                

                #if _ALPHATEST_ON
                    clip(col.a - _AlphaCutout);
                #endif

                if(_Visibillity < 0.5)
                {
                    discard;
                }

                
                return col;    
            }

            ENDHLSL
        }

                

        Pass
    {
        Name "ShadowCaster"

        Tags{"LightMode" = "ShadowCaster"}

            Cull Back
            HLSLPROGRAM
            #pragma prefer_hlslcc gles
            #pragma exclude_renderers d3d11_9x
            #pragma target 2.0

            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

           // GPU Instancing
            #pragma multi_compile_instancing

            #pragma shader_feature _ALPHATEST_ON
          
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            
           
            TEXTURE2D(_BaseMap);   SAMPLER(sampler_BaseMap);   
            CBUFFER_START(UnityPerMaterial)    

            float4 _BaseMap_ST;
            half4 _MaskTex_ST;
            
            float4 _Color,_FresnelColor;
            
            float _ReceiveShadowAmount;
            float _NormalStrength,_NormalGFlip;
            
            float _Metallic;
            float _Smoothness;
            float _AoStrength;


            float _AlphaCutout;

            float4 _EmissionColor;
            float _EmissionEdgeSmooth;
            float _EmissionIntensity;
            float _EmissionMode;


            float4 _MidColor, _ShadowColor, _ReflectColor, _ReflProbeColor;
            float _MidThreshold, _MidSmooth, _ShadowThreshold, _ShadowSmooth, _ReflectThreshold, _ReflectSmooth;
            float _SpecularThreshold, _SpecularSmooth, _SpecularIntensity ,_FresnelIntensity, _FresnelThreshold, _FresnelSmooth, _DirectionalFresnel, _MetalFresnelIntensity;
            float _ReflProbeIntensity, _MetalReflProbeIntensity, _ReflProbeBlurAmount;
            float _GIIntensity;

            float _UseGGX;

            float4 _SSSColor;
            float _Attenuation, _Distortion, _SSSPower, _SSSScale, _SSSAmbient;

            float _Visibillity;

            float4 _DyeingColor1, _DyeingColor2, _DyeingColor3, _DyeingColor4, _DyeingColor5, _DyeingColor6;
            float _DyeingAmount1, _DyeingAmount2, _DyeingAmount3, _DyeingAmount4, _DyeingAmount5, _DyeingAmount6;
            float _DyeingRoughness1, _DyeingRoughness2, _DyeingRoughness3, _DyeingRoughness4, _DyeingRoughness5, _DyeingRoughness6;
            float _DyeingArea1, _DyeingArea2, _DyeingArea3, _DyeingArea4, _DyeingArea5, _DyeingArea6;



            CBUFFER_END

            struct VertexInput
            {          
                float4 vertex : POSITION;
				float2 texcoord : TEXCOORD0;
                float4 normal : NORMAL;

                UNITY_VERTEX_INPUT_INSTANCE_ID  
            };
          
            struct VertexOutput
            {          
                float4 vertex : SV_POSITION;
				float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID          
                UNITY_VERTEX_OUTPUT_STEREO
  
            };

            VertexOutput ShadowPassVertex(VertexInput v)
            {
               VertexOutput o;
               UNITY_SETUP_INSTANCE_ID(v);
               UNITY_TRANSFER_INSTANCE_ID(v, o);                        
           
              float3 positionWS = TransformObjectToWorld(v.vertex.xyz);
              float3 normalWS   = TransformObjectToWorldNormal(v.normal.xyz);
              float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, _MainLightPosition.xyz));
              
              o.vertex = positionCS;
			  o.texcoord = v.texcoord;
             
              return o;
            }

            half4 ShadowPassFragment(VertexOutput i) : SV_TARGET
            {  
                UNITY_SETUP_INSTANCE_ID(i);
                
            #if _ALPHATEST_ON
                float4 diffuse = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.texcoord) ;
                clip(diffuse.a - _AlphaCutout);
            #endif

                if(_Visibillity < 0.5)
                {
                    discard;
                }
                return 0;
            }

            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags{"LightMode" = "DepthOnly"}

            ZWrite On
            ColorMask 0

            Cull Back

            HLSLPROGRAM
          
            #pragma prefer_hlslcc gles
            #pragma exclude_renderers d3d11_9x
            #pragma target 2.0
  
            // GPU Instancing
            #pragma multi_compile_instancing

            #pragma vertex vert
            #pragma fragment frag

            #pragma shader_feature _ALPHATEST_ON
              
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
              
            TEXTURE2D(_BaseMap);   SAMPLER(sampler_BaseMap);   
            CBUFFER_START(UnityPerMaterial)    

            float4 _BaseMap_ST;
            half4 _MaskTex_ST;
            
            float4 _Color,_FresnelColor;
            
            float _ReceiveShadowAmount;
            float _NormalStrength,_NormalGFlip;
            
            float _Metallic;
            float _Smoothness;
            float _AoStrength;


            float _AlphaCutout;

            float4 _EmissionColor;
            float _EmissionEdgeSmooth;
            float _EmissionIntensity;
            float _EmissionMode;


            float4 _MidColor, _ShadowColor, _ReflectColor, _ReflProbeColor;
            float _MidThreshold, _MidSmooth, _ShadowThreshold, _ShadowSmooth, _ReflectThreshold, _ReflectSmooth;
            float _SpecularThreshold, _SpecularSmooth, _SpecularIntensity ,_FresnelIntensity, _FresnelThreshold, _FresnelSmooth, _DirectionalFresnel, _MetalFresnelIntensity;
            float _ReflProbeIntensity, _MetalReflProbeIntensity, _ReflProbeBlurAmount;
            float _GIIntensity;

            float _UseGGX;

            float4 _SSSColor;
            float _Attenuation, _Distortion, _SSSPower, _SSSScale, _SSSAmbient;

            float _Visibillity;

            float4 _DyeingColor1, _DyeingColor2, _DyeingColor3, _DyeingColor4, _DyeingColor5, _DyeingColor6;
            float _DyeingAmount1, _DyeingAmount2, _DyeingAmount3, _DyeingAmount4, _DyeingAmount5, _DyeingAmount6;
            float _DyeingRoughness1, _DyeingRoughness2, _DyeingRoughness3, _DyeingRoughness4, _DyeingRoughness5, _DyeingRoughness6;
            float _DyeingArea1, _DyeingArea2, _DyeingArea3, _DyeingArea4, _DyeingArea5, _DyeingArea6;



            CBUFFER_END


              
            struct VertexInput
            {
                float4 vertex : POSITION;        
				float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct VertexOutput
            {          
                float4 vertex : SV_POSITION;
				float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID          
            };

            VertexOutput vert(VertexInput v)
            {
                VertexOutput o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                o.vertex = TransformWorldToHClip(TransformObjectToWorld(v.vertex.xyz));
				o.texcoord = v.texcoord;
                return o;
            }

            half4 frag(VertexOutput IN) : SV_TARGET
            {     
                
            #if _ALPHATEST_ON
                float4 diffuse = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.texcoord);
                clip(diffuse.a - _AlphaCutout);
            #endif         

                if(_Visibillity < 0.5)
                {
                    discard;
                }
                    
                return 0;
            }
            ENDHLSL
        }

        

            
    }
}






