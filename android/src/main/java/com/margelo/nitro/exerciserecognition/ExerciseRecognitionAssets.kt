package com.margelo.nitro.exerciserecognition

import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.NitroModules

@DoNotStrip
object ExerciseRecognitionAssets {
    @JvmStatic
    @DoNotStrip
    fun loadAssetAsString(assetName: String): String {
        val context = NitroModules.applicationContext
        if (context == null) {
            Log.e("exercise-recognition", "application context is null")
            return ""
        }

        return try {
            context.assets.open(assetName).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            Log.e("exercise-recognition", "failed to load asset $assetName: ${e.message}", e)
            ""
        }
    }
}
