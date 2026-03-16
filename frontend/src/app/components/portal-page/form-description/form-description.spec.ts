import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FormDescription } from './form-description';

describe('FormDescription', () => {
  let component: FormDescription;
  let fixture: ComponentFixture<FormDescription>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FormDescription],
    }).compileComponents();

    fixture = TestBed.createComponent(FormDescription);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
