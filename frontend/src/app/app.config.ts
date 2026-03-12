import { ApplicationConfig } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideRouter } from '@angular/router';
import { provideTranslateService, provideTranslateLoader } from '@ngx-translate/core';
import { TranslateHttpLoader, provideTranslateHttpLoader } from '@ngx-translate/http-loader';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient(),
    provideTranslateHttpLoader({ prefix: '/assets/i18n/', suffix: '.json' }),
    provideTranslateService({
      defaultLanguage: 'en',
    }),
    provideTranslateLoader(TranslateHttpLoader),
  ]
};
