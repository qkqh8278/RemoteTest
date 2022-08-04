Shader "ADA/Character/StylizedHairShader"
{
    Properties
    {
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
        
        _DyeingColor("Dyeing Color ", Color) = (1,1,1,1)
        _DyeingAmount("Dyeing Amount ", Range(0,1)) = 1

    
        [Space(20)]
        [Header(Stylized Reflection)]
        [Space(10)]
        [NoScaleOffset] _MaskTex ("Mask Texture ( R:Smoothness, G:Metallic, B:AO)", 2D) = "white" {}
        _Smoothness ("Smoothness", Range(0,1)) = 1
        _Metallic ("Metallic ", Range(0,1)) = 1
        
        _AoStrength("AO Strength", Range(0, 1)) = 1.0

        [Space(10)]
        _EmissionColor ("Emission Color", Color) = (1,1,1,1)
        _EmissionEdgeSmooth ("Emission Edge Smooth ", float) = 1
        _EmissionIntensity ("Emission Intensity", float) = 1

        
        [Space(10)]

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

        [Space(10)]
        [Header(Hair)]
        [Space(10)]
        _AnisoTex ("Aniso Shift Texture (R)", 2D) = "gray" {}
        //_UVCon ("UV Con", Range(0,1)) = 0
        //_RotateTangent("Rotate Tangent", Vector) = (0,0,0,0)
        
        [Space(10)]
        _LowSpecularColor("Low Specular Color", Color) = (0.5,0.5,0.5,1)
        _LowSpecularPosition("Low Specular Position Offset", Range(-10,10) ) = 0
        
        _LowSpecularPower ("Low Specular Power", Range(0,1)) = 0.02
        _LowSpecularIntensity ("Low Specular Intensity", float) = 1
        _LowShiftMin("Low Shift Min", float) = 0
        _LowShiftMax("Low Shift Max", float) = 1
        [Space(10)]
        _HighSpecularColor("High Specular Color", Color) = (1,1,1,1)
        _HighSpecularPosition("High Specular Position Offset", Range(-10,10) ) = 0
        
        _HighSpecularPower ("High Specular Power", Range(0,1)) = 0.0025
        _HighSpecularIntensity ("High Specular Intensity", float) = 1
        _HighShiftMin("High Shift Min", float) = 0
        _HighShiftMax("High Shift Max", float) = 1

   
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
            "RenderType"="Transparent"
            "Queue"="Transparent+0"
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
            Blend SrcAlpha OneMinusSrcAlpha
            Zwrite Off


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
                        
            
            

            TEXTURE2D(_BaseMap);   SAMPLER(sampler_BaseMap);   
            TEXTURE2D(_NormalTex);   SAMPLER(sampler_NormalTex);  
            TEXTURE2D(_MaskTex);   SAMPLER(sampler_MaskTex); 
            TEXTURE2D(_AnisoTex); SAMPLER(sampler_AnisoTex); 

            CBUFFER_START(UnityPerMaterial)    

            float4 _BaseMap_ST;
            half4 _MaskTex_ST, _AnisoTex_ST;
            
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

            float4 _RotateTangent;
            float _UVCon;

            float4 _LowSpecularColor;
            float _LowSpecularIntensity;
            float _LowSpecularPower;
            
            float _LowSpecularPosition;
            float _LowShiftMax;
            float _LowShiftMin;

            float4 _HighSpecularColor;
            float _HighSpecularIntensity;
            float _HighSpecularPower;
            
            float _HighSpecularPosition;
            float _HighShiftMax;
            float _HighShiftMin;

            float4 _DyeingColor;
            float _DyeingAmount;


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
                float3 Tan = T.xyz;  
                float3 Nor = N.xyz;
                float3 H = normalize(L+V);
                float3 B = normalize(cross( Tan ,Nor) * T.w);
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

            half3 DirectBDRFCustom(BRDFData brdfData, half3 normalWS, half3 lightDirectionWS, half3 viewDirectionWS)
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
                half3 color = specularTerm * brdfData.specular * max(0,_SpecularIntensity) + brdfData.diffuse; //LinearStep( _SpecularThreshold - _SpecularSmooth, _SpecularThreshold + _SpecularSmooth, specularTerm  ) * brdfData.specular * max(0,_SpecularIntensity) + brdfData.diffuse;
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



            half3 LightingPhysicallyBasedCustom(BRDFData brdfData, half3 radiance, half3 lightDirectionWS, half3 normalWS, half3 viewDirectionWS, half3 positionWS)
            {             
            
                return DirectBDRFCustom(brdfData, normalWS, lightDirectionWS, viewDirectionWS) * radiance;
            }

            half3 LightingPhysicallyBasedCustom(BRDFData brdfData, half3 radiance, Light light, half3 normalWS, half3 viewDirectionWS, half3 positionWS)
            {
                return LightingPhysicallyBasedCustom(brdfData, radiance, light.direction, normalWS, viewDirectionWS, positionWS );
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



            half4 UniversalFragmentPBRCustom(InputData inputData, half3 albedo, half metallic, half3 specular, half smoothness, half occlusion, half3 emission, half alpha)
            {
                BRDFData brdfData;
                InitializeBRDFData(albedo, metallic, specular, smoothness, alpha, brdfData);
                
                inputData.shadowCoord = TransformWorldToShadowCoord(inputData.positionWS);
                Light mainLight = GetMainLight(inputData.shadowCoord);


                

                
                float3 radiance = CalculateRadiance(mainLight, inputData.normalWS, float3(0, 0, 0));


                MixRealtimeAndBakedGI(mainLight, inputData.normalWS, inputData.bakedGI, half4(0, 0, 0, 0));

                float ndotl = lerp(1, LinearStep( _ShadowThreshold - _ShadowSmooth, _ShadowThreshold + _ShadowSmooth,  dot(mainLight.direction, inputData.normalWS) * 0.5 + 0.5), _DirectionalFresnel);

                half3 color = GlobalIlluminationCustom(brdfData, radiance, inputData.bakedGI, occlusion, inputData.normalWS, inputData.viewDirectionWS, metallic, ndotl);
                color += LightingPhysicallyBasedCustom(brdfData, radiance, mainLight, inputData.normalWS, inputData.viewDirectionWS, inputData.positionWS);

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
                output.positionCS = TransformObjectToHClip(input.vertex.xyz);//, float3(input.texcoord.xy,1), _UVCon) );//vertexInput.positionCS;
                output.fogFactorAndVertexLight = half4(ComputeFogFactor(vertexInput.positionCS.z), vertexLight);
                
                


                return output;
            }

            half4 frag (v2f i) : SV_Target
            {

                UNITY_SETUP_INSTANCE_ID(i);
                
               
                float3 NormalTS = UnpackNormal( SAMPLE_TEXTURE2D(_NormalTex, sampler_NormalTex, i.uv) ) * float3(_NormalStrength, _NormalStrength * lerp(1,-1,_NormalGFlip), 1);
                
                float4 albedo = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv) * _Color;

                albedo.rgb = lerp(albedo.rgb, BlendOverlay(albedo.rgb, _DyeingColor.rgb), _DyeingAmount);

                Light mainLight = GetMainLight();

                float4 col;

                InputData inputData;
                InitializeInputData(i, NormalTS, inputData);
                i.shadowCoord = TransformWorldToShadowCoord(i.positionWS);

                float4 mask = SAMPLE_TEXTURE2D(_MaskTex, sampler_MaskTex, i.uv);

                float metallic = mask.g * _Metallic;
                float smoothness = mask.r * _Smoothness;
                float occlusion =  lerp( 1 , mask.b , _AoStrength);
                
                

                float3 emissive = 0;


                half smoothShadow = LinearStep ( _ShadowThreshold - _ShadowSmooth, _ShadowThreshold + _ShadowSmooth, dot( mainLight.direction, inputData.normalWS )*0.5+0.5 ) * (lerp(1,mainLight.distanceAttenuation * mainLight.shadowAttenuation,_ReceiveShadowAmount) ) ;

                col = UniversalFragmentPBRCustom(inputData, albedo.rgb, metallic, 0.5 , smoothness , occlusion, emissive.rgb, albedo.a);


                float4 shiftTex = SAMPLE_TEXTURE2D(_AnisoTex, sampler_AnisoTex, i.uv * _AnisoTex_ST.xy + _AnisoTex_ST.zw);
                
                float lowShift = lerp(_LowShiftMin, _LowShiftMax, shiftTex.r);
                float highShift = lerp(_HighShiftMin, _HighShiftMax, shiftTex.r);

                float lowSpecular = StrandSpecular( i.tangentWS, i.viewDirWS, normalize(mainLight.direction + float3(0,_LowSpecularPosition,0)), inputData.normalWS, lowShift-0.5, _LowSpecularPower, 1);
                float highSpecular = StrandSpecular(i.tangentWS, i.viewDirWS, normalize(mainLight.direction + float3(0,_HighSpecularPosition,0)), inputData.normalWS, highShift-0.5, _HighSpecularPower, 1);
                float3 lowSpecularColor = lowSpecular * _LowSpecularColor.rgb ;
                float3 highSpecularColor = highSpecular * _HighSpecularColor.rgb ;
                col.rgb += ((lowSpecularColor * _LowSpecularIntensity) + (highSpecularColor * _HighSpecularIntensity)) * occlusion * smoothShadow ;



                col.rgb = MixFog(col.rgb, inputData.fogCoord);
                col.a *= _Color.a;

                #if _ALPHATEST_ON
                    clip(col.a - _AlphaCutout);
                #endif

                
                return col;    
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
            half4 _MaskTex_ST, _AnisoTex_ST;
            
            float4 _Color, _FresnelColor;
            
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

            float4 _RotateTangent;

            float4 _LowSpecularColor;
            float _LowSpecularIntensity;
            float _LowSpecularPower;
            
            float _LowSpecularPosition;
            float _LowShiftMax;
            float _LowShiftMin;

            float4 _HighSpecularColor;
            float _HighSpecularIntensity;
            float _HighSpecularPower;
            
            float _HighSpecularPosition;
            float _HighShiftMax;
            float _HighShiftMin;

            float4 _DyeingColor;
            float _DyeingAmount;


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
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "PreZ"
            Tags{"LightMode" = "PreZ"}

            ZWrite On
            ColorMask 0

            Cull Front

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
            
            float4 _Color, _FresnelColor;
            
            float _ReceiveShadowAmount;
            float _NormalStrength;
            
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

            float4 _LowSpecularColor;
            float _LowSpecularIntensity;
            float _LowSpecularPower;
            
            float _LowSpecularPosition;
            float _LowShiftMax;
            float _LowShiftMin;

            float4 _HighSpecularColor;
            float _HighSpecularIntensity;
            float _HighSpecularPower;
            
            float _HighSpecularPosition;
            float _HighShiftMax;
            float _HighShiftMin;

            float4 _DyeingColor;
            float _DyeingAmount;


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

            float3 BlendOverlay(float3 base, float3 blend)
            {
                float3 check = step(0.5, base);
                float3 result = check * (half3(1, 1, 1) - ((half3(1, 1, 1) - 2 * (base - 0.5f)) * (1 - blend)));
                result += (1 - check) * (2 * base) * blend;
                return result;
            }

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
                return 0;
            }
            ENDHLSL
        }


        

            
    }
}






